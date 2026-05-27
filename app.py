"""Streamlit-based user interface and pipeline integration.

Allows users to upload images or take snapshots via webcam,
visualize the image processing pipeline (contours, binary masks),
calibrate HSV threshold ranges live, inspect feature vectors,
log prediction history, and play against the AI in a game mode.

Author: Dawid Wiśniowski
"""

from __future__ import annotations

import datetime
import random
import sys
from pathlib import Path
from typing import Optional, Tuple

import cv2  # type: ignore
import numpy as np
import streamlit as st
from PIL import Image

# Pipeline imports
from contour_detection import (
    GREEN_BG_HSV,
    apply_morphology,
    filter_contours_by_area,
    find_contours,
    get_main_contour,
    invert_mask,
)
from feature_extraction import FEATURE_NAMES, extract_feature_vector
from feature_normalization import CLASS_NAMES
from image_preprocessing import HSVRange, preprocess_to_hsv_mask
from predict import load_model

# ---------------------------------------------------------------------------
# Page configuration & Styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Klasyfikator AI Kamień-Papier-Nożyce",
    page_icon="✂️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern glassmorphism aesthetic
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;800&family=Outfit:wght@300;500;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
}

.main-title {
    font-family: 'Montserrat', sans-serif;
    font-weight: 800;
    text-align: center;
    background: linear-gradient(135deg, #6B73FF 0%, #000DFF 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 3rem;
    padding-bottom: 0.5rem;
    margin-bottom: 0.2rem;
}

.main-subtitle {
    font-family: 'Outfit', sans-serif;
    font-weight: 300;
    text-align: center;
    color: #555555;
    font-size: 1.2rem;
    margin-bottom: 2.5rem;
}

.section-card {
    background-color: #F8F9FA;
    padding: 20px;
    border-radius: 16px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    border: 1px solid #EAEAEA;
    margin-bottom: 20px;
}

.result-card {
    text-align: center;
    padding: 25px;
    border-radius: 20px;
    color: white;
    font-weight: 700;
    font-size: 2rem;
    margin-bottom: 20px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.1);
}

.result-rock {
    background: linear-gradient(135deg, #FF6B6B 0%, #FF8E53 100%);
}

.result-paper {
    background: linear-gradient(135deg, #4E65FF 0%, #92EFFD 100%);
}

.result-scissors {
    background: linear-gradient(135deg, #11998E 0%, #38EF7D 100%);
}

.result-none {
    background: linear-gradient(135deg, #7F8C8D 0%, #95A5A6 100%);
}

.game-title {
    font-family: 'Montserrat', sans-serif;
    font-size: 1.8rem;
    font-weight: 700;
    margin-bottom: 15px;
    color: #2C3E50;
    border-bottom: 2px solid #EAEAEA;
    padding-bottom: 8px;
}

.game-vs-badge {
    font-size: 1.5rem;
    font-weight: 800;
    color: #7F8C8D;
    display: flex;
    align-items: center;
    justify-content: center;
    height: 100%;
}
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

if "game_scores" not in st.session_state:
    st.session_state.game_scores = {"player": 0, "ai": 0, "draws": 0}

if "game_round" not in st.session_state:
    st.session_state.game_round = 0

if "game_log" not in st.session_state:
    st.session_state.game_log = []


# ---------------------------------------------------------------------------
# Helper Functions & Prediction Pipeline
# ---------------------------------------------------------------------------
@st.cache_resource
def get_pipeline_resources() -> Tuple[Optional[object], Optional[object], Optional[str]]:
    """Helper to safely load models/scalers."""
    try:
        model, scaler = load_model(
            model_path=Path("models/knn_model.pkl"),
            scaler_path=Path("models/scaler.pkl"),
        )
        return model, scaler, None
    except Exception as e:
        return None, None, str(e)


def run_prediction_pipeline(
    img_rgb: np.ndarray,
    model: object,
    scaler: object,
    hsv_range: HSVRange,
    morph_kernel_size: int,
    min_contour_area: float,
) -> Tuple[Optional[str], Optional[np.ndarray], np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Runs the computer vision processing and k-NN classifier on the image.

    Center-crops the input image to a 3:2 aspect ratio and resizes it to 300x200
    internally to prevent distortion of the hand shapes.

    Args:
        img_rgb: Input RGB image (H, W, 3).
        model: KNeighborsClassifier.
        scaler: StandardScaler/MinMaxScaler.
        hsv_range: Selected HSV background thresholds.
        morph_kernel_size: Size of morphology kernel.
        min_contour_area: Minimum area to keep a contour.

    Returns:
        Tuple: (label, probabilities, binary_mask, main_contour, feature_vector)
    """
    original_h, original_w = img_rgb.shape[:2]
    
    # 1. Crop to 3:2 aspect ratio (1.5) to prevent distortion
    target_aspect = 1.5
    current_aspect = original_w / original_h
    
    if current_aspect > target_aspect:
        # Image is too wide - crop the sides
        new_w = int(original_h * target_aspect)
        start_x = (original_w - new_w) // 2
        cropped_rgb = img_rgb[:, start_x:start_x + new_w]
        crop_x, crop_y = start_x, 0
        crop_w, crop_h = new_w, original_h
    elif current_aspect < target_aspect:
        # Image is too tall - crop top/bottom
        new_h = int(original_w / target_aspect)
        start_y = (original_h - new_h) // 2
        cropped_rgb = img_rgb[start_y:start_y + new_h, :]
        crop_x, crop_y = 0, start_y
        crop_w, crop_h = original_w, new_h
    else:
        cropped_rgb = img_rgb
        crop_x, crop_y = 0, 0
        crop_w, crop_h = original_w, original_h

    # Resize cropped image to dataset size (300x200)
    resized_rgb = cv2.resize(cropped_rgb, (300, 200))

    # 2. Image preprocessing (Gaussian blur + HSV thresholding)
    bg_mask = preprocess_to_hsv_mask(resized_rgb, hsv_range, blur_kernel=(5, 5))

    # 3. Mask Inversion (green backround is black, hand is white)
    hand_mask = invert_mask(bg_mask)

    # 4. Clean up noise using Morphological Operations
    cleaned_mask = apply_morphology(hand_mask, kernel_size=morph_kernel_size)

    # 5. Contour Detection and filtering
    contours = find_contours(cleaned_mask)
    filtered_contours = filter_contours_by_area(contours, min_area=min_contour_area)
    main_contour = get_main_contour(filtered_contours)

    if main_contour is None:
        # Resize cleaned mask back to cropped size and insert into original black frame
        cropped_mask = cv2.resize(cleaned_mask, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)
        disp_mask = np.zeros((original_h, original_w), dtype=np.uint8)
        disp_mask[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w] = cropped_mask
        return None, None, disp_mask, None, None

    # 6. Extract Feature Vector (on 300x200 scale)
    feat = extract_feature_vector(main_contour)

    # 7. Normalize and Predict
    feat_2d = feat.reshape(1, -1)
    feat_scaled = scaler.transform(feat_2d)

    label_idx = int(model.predict(feat_scaled)[0])
    proba = model.predict_proba(feat_scaled)[0]
    label_name = CLASS_NAMES[label_idx]

    # Scale contour back to original resolution, accounting for crop offset
    scale_x = crop_w / 300.0
    scale_y = crop_h / 200.0
    scaled_contour = main_contour.copy()
    scaled_contour[:, 0, 0] = np.round(main_contour[:, 0, 0] * scale_x) + crop_x
    scaled_contour[:, 0, 1] = np.round(main_contour[:, 0, 1] * scale_y) + crop_y

    # Scale mask back to original resolution with padding
    cropped_mask = cv2.resize(cleaned_mask, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)
    disp_mask = np.zeros((original_h, original_w), dtype=np.uint8)
    disp_mask[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w] = cropped_mask

    return label_name, proba, disp_mask, scaled_contour, feat


def get_gesture_emoji(gesture: Optional[str]) -> str:
    if gesture == "rock":
        return "🪨 KAMIEŃ"
    elif gesture == "paper":
        return "📄 PAPIER"
    elif gesture == "scissors":
        return "✂️ NOŻYCE"
    return "❓ NIEZNANY"


def evaluate_rps(p_choice: str, ai_choice: str) -> str:
    """Return winner description: 'player', 'ai', or 'draw'."""
    if p_choice == ai_choice:
        return "draw"
    wins = {
        "rock": "scissors",
        "scissors": "paper",
        "paper": "rock",
    }
    if wins[p_choice] == ai_choice:
        return "player"
    return "ai"


# ---------------------------------------------------------------------------
# Sidebar Configuration Panel
# ---------------------------------------------------------------------------
st.sidebar.image(
    "https://images.unsplash.com/photo-1612196808214-b8e1d6145a8c?auto=format&fit=crop&w=400&q=80",
    caption="Klasyfikator Wizji Komputerowej",
    width="stretch",
)

st.sidebar.title("🛠️ Panel Kalibracji")

st.sidebar.write("Skonfiguruj parametry segmentacji, aby odizolować dłoń od tła.")

# HSV Threshold Sliders
st.sidebar.subheader("Zakres tła HSV")
h_low, h_high = st.sidebar.slider("Odcień (Hue - tło)", 0, 179, (35, 85))
s_low, s_high = st.sidebar.slider("Nasycenie (Saturation)", 0, 255, (50, 255))
v_low, v_high = st.sidebar.slider("Jasność (Value)", 0, 255, (50, 255))
hsv_config = HSVRange(lower=(h_low, s_low, v_low), upper=(h_high, s_high, v_high))

# Morphological Operations Slider
st.sidebar.subheader("Ustawienia morfologii")
morph_size = st.sidebar.slider(
    "Rozmiar jądra (Usuwanie szumów)",
    min_value=3,
    max_value=15,
    value=5,
    step=2,
    help="Tylko liczby nieparzyste. Usuwa małe drobiny szumu z tła i pierwszego planu.",
)

# Minimum Contour Area Slider
min_area = st.sidebar.slider(
    "Min. pole powierzchni dłoni (px)",
    min_value=100,
    max_value=10000,
    value=1000,
    step=100,
    help="Minimalne pole powierzchni konturu, aby został uznany za gest dłoni.",
)

# Reset defaults button
if st.sidebar.button("Resetuj do domyślnego zielonego tła"):
    st.rerun()

st.sidebar.markdown("---")
# Loading model resources
model, scaler, err_msg = get_pipeline_resources()
if model is None:
    st.sidebar.error(f"Nie można załadować zasobów modelu: {err_msg}")
    st.stop()
else:
    st.sidebar.success("Model załadowany pomyślnie.")


# ---------------------------------------------------------------------------
# Main Layout
# ---------------------------------------------------------------------------
st.markdown("<h1 class='main-title'>Kamień, Papier, Nożyce</h1>", unsafe_allow_html=True)
st.markdown(
    "<div class='main-subtitle'>Interaktywny klasyfikator gestów k-NN i wizualizator potoku przetwarzania w czasie rzeczywistym</div>",
    unsafe_allow_html=True,
)

tab1, tab2, tab3 = st.tabs(["🔍 Obszar roboczy", "🎮 Gra z AI", "📜 Historia i statystyki"])

# ===========================================================================
# TAB 1: Live Workspace
# ===========================================================================
with tab1:
    col_input, col_results = st.columns([1, 1])

    with col_input:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.subheader("📸 Źródło wejściowe gestu")
        source_options = ["camera", "upload"]
        source_translation = {
            "camera": "Obraz z kamery",
            "upload": "Prześlij plik graficzny"
        }
        source = st.radio(
            "Wybierz metodę wprowadzania",
            options=source_options,
            format_func=lambda x: source_translation.get(x, x),
            label_visibility="collapsed",
        )

        img_bgr: Optional[np.ndarray] = None

        if source == "camera":
            camera_image = st.camera_input("Zrób zdjęcie gestu dłoni")
            if camera_image:
                pil_image = Image.open(camera_image)
                # Convert PIL (RGB) to numpy RGB
                img_rgb = np.array(pil_image.convert("RGB"))
                # Pipeline expects BGR internally for predict.py compatibility, let's keep RGB for displaying
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        else:
            uploaded_file = st.file_uploader(
                "Prześlij plik graficzny PNG/JPG (najlepiej o rozdzielczości 300x200)",
                type=["png", "jpg", "jpeg"],
            )
            if uploaded_file:
                pil_image = Image.open(uploaded_file)
                img_rgb = np.array(pil_image.convert("RGB"))
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                st.image(img_rgb, caption="Przesłany oryginalny obraz", width="stretch")

        st.markdown("</div>", unsafe_allow_html=True)

    with col_results:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.subheader("⚡ Wyniki predykcji")

        if img_bgr is not None:
            # Run the process pipeline
            label, probas, mask, contour, feat = run_prediction_pipeline(
                img_rgb, model, scaler, hsv_config, morph_size, min_area
            )

            if label is not None and probas is not None and feat is not None:
                # Add to history
                current_time = datetime.datetime.now().strftime("%H:%M:%S")
                # Avoid adding duplicates from rapid redraws
                last_time = st.session_state.history[-1].get("time") if st.session_state.history else None
                if not st.session_state.history or last_time != current_time:
                    st.session_state.history.append(
                        {
                            "time": current_time,
                            "source": "camera" if source == "camera" else "file",
                            "prediction": label,
                            "confidence": f"{float(np.max(probas)):.1%}",
                        }
                    )

                # Render Card based on predicted label
                emoji_label = get_gesture_emoji(label)
                max_idx = int(np.argmax(probas))
                confidence_pct = probas[max_idx] * 100

                card_class = f"result-card result-{label}"
                st.markdown(
                    f"<div class='{card_class}'>{emoji_label} ({confidence_pct:.1f}%)</div>",
                    unsafe_allow_html=True,
                )

                # Draw Confidence Distributions
                st.write("**Rozkład pewności klasyfikacji**")
                for c_idx, c_name in enumerate(CLASS_NAMES):
                    conf = float(probas[c_idx])
                    emoji_c = "🪨" if c_name == "rock" else "📄" if c_name == "paper" else "✂️"
                    translated_name = "Kamień" if c_name == "rock" else "Papier" if c_name == "paper" else "Nożyce"
                    col_lbl, col_bar = st.columns([1, 4])
                    col_lbl.write(f"{emoji_c} {translated_name}")
                    col_bar.progress(conf)

            else:
                st.markdown(
                    "<div class='result-card result-none'>❌ NIE WYKRYTO DŁONI</div>",
                    unsafe_allow_html=True,
                )
                st.warning(
                    "Algorytm segmentacji nie mógł zidentyfikować poprawnego konturu dłoni. "
                    "Dostosuj suwaki w Panelu Kalibracji (Odcień, Nasycenie, Jasność), "
                    "aby dopasować je do oświetlenia i tła."
                )
        else:
            st.info("Zrób zdjęcie lub prześlij obraz, aby rozpocząć klasyfikację.")

        st.markdown("</div>", unsafe_allow_html=True)

    # Pipeline Visualization Tabs
    if img_bgr is not None:
        st.markdown("---")
        st.subheader("🖼️ Wizualizator etapów przetwarzania (Pipeline)")
        tab_v1, tab_v2, tab_v3 = st.tabs(
            ["1. Wyizolowany kontur", "2. Maska binarna (Usuwanie tła)", "3. Wyekstrahowany wektor cech"]
        )

        with tab_v1:
            col_v1_img, col_v1_desc = st.columns([2, 1])
            with col_v1_img:
                # Draw contour on original image
                disp_rgb = img_rgb.copy()
                if contour is not None:
                    cv2.drawContours(disp_rgb, [contour], -1, (255, 0, 0), 2)
                    # Put label overlay
                    if label is not None:
                        # Find top left of bounding rect
                        x, y, w, h = cv2.boundingRect(contour)
                        translated_label = "KAMIEN" if label == "rock" else "PAPIER" if label == "paper" else "NOZYCE"
                        cv2.putText(
                            disp_rgb,
                            f"{translated_label} {np.max(probas)*100:.0f}%",
                            (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 0, 0),
                            2,
                        )
                st.image(disp_rgb, caption="Wykryty kontur dłoni (Czerwony)", width="stretch")
            with col_v1_desc:
                st.write("### Etap 4: Detekcja konturów")
                st.write(
                    "Wyszukiwanie konturów na oczyszczonej masce binarnej za pomocą `cv2.findContours` z biblioteki OpenCV. "
                    "Największy kontur spełniający minimalny próg pola powierzchni jest izolowany jako kształt dłoni."
                )
                if contour is not None:
                    st.write(f"- **Punkty konturu:** {len(contour)}")
                    st.write(f"- **Prostokąt otaczający (Bounding Box):** {cv2.boundingRect(contour)}")

        with tab_v2:
            col_v2_img, col_v2_desc = st.columns([2, 1])
            with col_v2_img:
                # Display binary mask
                st.image(mask, caption="Przetworzona maska binarna", clamp=True, width="stretch")
            with col_v2_desc:
                st.write("### Etapy 2 i 3: Segmentacja i morfologia")
                st.write(
                    "Najpierw system konwertuje obraz do przestrzeni barw HSV i maskuje tło "
                    "na podstawie ustawionego zakresu HSV. Następnie stosowane są operacje "
                    "morfologiczne w celu usunięcia szumów:"
                )
                st.write(
                    f"1. **Erozja i dylacja** do eliminacji małych pojedynczych białych punktów.\n"
                    f"2. **Zamknięcie i otwarcie** (Closing & Opening) z eliptycznym jądrem o rozmiarze "
                    f"`{morph_size}x{morph_size}` w celu usunięcia dziur wewnątrz dłoni i wygładzenia krawędzi."
                )

        with tab_v3:
            st.write("### Etapy 5 i 6: Cechy geometryczne")
            st.write(
                "Z obrysu dłoni obliczanych jest łącznie **12 deskryptorów matematycznych** tworzących "
                "wektor cech. Są one normalizowane i porównywane za pomocą odległości euklidesowej w algorytmie k-NN."
            )
            if feat is not None:
                # Translate names for display
                feature_names_pl = {
                    "hu_0": "Hu moment 0", "hu_1": "Hu moment 1", "hu_2": "Hu moment 2",
                    "hu_3": "Hu moment 3", "hu_4": "Hu moment 4", "hu_5": "Hu moment 5",
                    "hu_6": "Hu moment 6",
                    "area": "Pole powierzchni (Area)",
                    "perimeter": "Obwód (Perimeter)",
                    "aspect_ratio": "Proporcje (Aspect Ratio)",
                    "convexity": "Wypukłość (Convexity)",
                    "circularity": "Okrągłość (Circularity)"
                }
                features_dict = {feature_names_pl.get(k, k): v for k, v in zip(FEATURE_NAMES, feat)}

                col_hu, col_geom = st.columns(2)
                with col_hu:
                    st.write("**Niezmienniki momentowe Hu (po transformacji logarytmicznej)**")
                    st.write(
                        "Momenty Hu dostarczają cech kształtu niezmiennych względem skali, obrotu i przesunięcia."
                    )
                    hu_data = {k: v for k, v in features_dict.items() if k.startswith("Hu moment")}
                    st.dataframe(hu_data, use_container_width=True)

                with col_geom:
                    st.write("**Deskryptory geometryczne i kształtu**")
                    st.write("Dodatkowe cechy opisujące fizyczne wymiary i proporcje obrysu dłoni.")
                    geom_keys = [
                        "Pole powierzchni (Area)", "Obwód (Perimeter)", 
                        "Proporcje (Aspect Ratio)", "Wypukłość (Convexity)", "Okrągłość (Circularity)"
                    ]
                    geom_data = {k: features_dict[k] for k in geom_keys}
                    st.dataframe(geom_data, use_container_width=True)


# ===========================================================================
# TAB 2: Play VS AI (Game Mode)
# ===========================================================================
with tab2:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown("<div class='game-title'>🎮 Arena Człowiek vs. Maszyna</div>", unsafe_allow_html=True)

    # Scoreboard
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    col_s1.metric("Twój wynik", st.session_state.game_scores["player"])
    col_s2.metric("Wynik AI", st.session_state.game_scores["ai"])
    col_s3.metric("Remisy", st.session_state.game_scores["draws"])
    col_s4.metric("Rozegrane rundy", st.session_state.game_round)

    col_btn1, col_btn2 = st.columns([1, 4])
    if col_btn1.button("Resetuj wynik gry"):
        st.session_state.game_scores = {"player": 0, "ai": 0, "draws": 0}
        st.session_state.game_round = 0
        st.session_state.game_log = []
        st.success("Wynik zresetowany!")
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    col_game_cam, col_game_vs, col_game_ai = st.columns([2, 1, 2])

    player_gesture: Optional[str] = None
    player_image_rgb: Optional[np.ndarray] = None

    with col_game_cam:
        st.subheader("Twoja dłoń")
        game_cam = st.camera_input("Zrób zdjęcie swojego gestu!", key="game_webcam")
        if game_cam:
            pil_image = Image.open(game_cam)
            player_image_rgb = np.array(pil_image.convert("RGB"))

            # Predict player gesture
            label, probas, _, _, _ = run_prediction_pipeline(
                player_image_rgb, model, scaler, hsv_config, morph_size, min_area
            )
            player_gesture = label

            if player_gesture:
                st.success(f"Wykryto: {get_gesture_emoji(player_gesture)}")
            else:
                st.error("Nie wykryto dłoni. Dostosuj suwaki w panelu bocznym.")

    with col_game_ai:
        st.subheader("Dłoń AI")
        ai_placeholder = st.empty()
        ai_placeholder.info("Oczekiwanie na ruch Gracza...")

    with col_game_vs:
        st.markdown("<div class='game-vs-badge'>VS</div>", unsafe_allow_html=True)
        outcome_placeholder = st.empty()

    if player_gesture and player_image_rgb is not None:
        # Trigger computer turn
        ai_choice = random.choice(CLASS_NAMES)

        # Show AI hand
        ai_emoji = get_gesture_emoji(ai_choice)
        ai_placeholder.markdown(
            f"<div class='result-card result-{ai_choice}' style='font-size:2.5rem; padding: 45px 10px;'>{ai_emoji}</div>",
            unsafe_allow_html=True,
        )

        # Evaluate round
        winner = evaluate_rps(player_gesture, ai_choice)
        st.session_state.game_round += 1

        # Save to game logs
        st.session_state.game_log.append(
            {
                "round": st.session_state.game_round,
                "player": player_gesture,
                "ai": ai_choice,
                "result": winner,
            }
        )

        if winner == "player":
            st.session_state.game_scores["player"] += 1
            outcome_placeholder.balloons()
            outcome_placeholder.success("### WYGRYWASZ! 🎉")
        elif winner == "ai":
            st.session_state.game_scores["ai"] += 1
            outcome_placeholder.error("### AI WYGRYWA! 🤖")
        else:
            st.session_state.game_scores["draws"] += 1
            outcome_placeholder.info("### REMIS! 🤝")

    # Render round history logs
    if st.session_state.game_log:
        st.markdown("---")
        st.subheader("Historia meczów")
        display_game_log = []
        for log in st.session_state.game_log[::-1]:
            round_val = log.get("round") if "round" in log else log.get("Runda", 0)
            
            p_val = log.get("player")
            if p_val is None:
                p_val_pl = log.get("Gracz", "")
            else:
                p_val_pl = get_gesture_emoji(p_val)
                
            ai_val = log.get("ai")
            if ai_val is None:
                ai_val_pl = log.get("AI", "")
            else:
                ai_val_pl = get_gesture_emoji(ai_val)
                
            res_val = log.get("result")
            if res_val is None:
                res_val_pl = log.get("Wynik", "")
            else:
                if res_val == "player":
                    res_val_pl = "WYGRANA GRACZA 🏆"
                elif res_val == "ai":
                    res_val_pl = "WYGRANA AI 🤖"
                else:
                    res_val_pl = "REMIS 🤝"
            
            display_game_log.append({
                "Runda": round_val,
                "Gracz": p_val_pl,
                "AI": ai_val_pl,
                "Wynik": res_val_pl
            })
        st.dataframe(
            display_game_log,
            use_container_width=True,
        )


# ===========================================================================
# TAB 3: History & Analytics
# ===========================================================================
with tab3:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("📜 Historia predykcji (Bieżąca sesja)")

    if st.session_state.history:
        col_c1, col_c2 = st.columns([1, 4])
        if col_c1.button("Wyczyść historię predykcji"):
            st.session_state.history = []
            st.success("Historia wyczyszczona!")
            st.rerun()

        # Display history as table
        display_history = []
        for h in st.session_state.history[::-1]:
            # Handle potential old keys defensively
            t_val = h.get("time") if "time" in h else h.get("Czas", "")
            s_val = h.get("source") if "source" in h else h.get("Źródło", "")
            p_val = h.get("prediction") if "prediction" in h else h.get("Predykcja", "")
            c_val = h.get("confidence") if "confidence" in h else h.get("Pewność", "")
            
            # Map source to Polish for display
            if s_val == "camera":
                s_val_pl = "Obraz z kamery"
            elif s_val == "file":
                s_val_pl = "Prześlij plik graficzny"
            else:
                s_val_pl = s_val
            
            display_history.append({
                "Czas": t_val,
                "Źródło": s_val_pl,
                "Predykcja": get_gesture_emoji(p_val) if p_val in CLASS_NAMES else p_val,
                "Pewność": c_val
            })
        st.dataframe(display_history, use_container_width=True)

        # Draw a bar chart showing counts of predicted gestures
        preds = [h.get("prediction") if "prediction" in h else h.get("Predykcja", "") for h in st.session_state.history]
        counts = {
            "Kamień": preds.count("rock") + preds.count("🪨 KAMIEŃ"),
            "Papier": preds.count("paper") + preds.count("📄 PAPIER"),
            "Nożyce": preds.count("scissors") + preds.count("✂️ NOŻYCE")
        }

        st.markdown("### Rozkład gestów w sesji")
        st.bar_chart(counts)

    else:
        st.info("Brak predykcji zapisanych w tej sesji.")
    st.markdown("</div>", unsafe_allow_html=True)
