import streamlit as st
from poseperfect_ai.preprocessing.image_preprocessor import preprocess_for_static_analysis
from poseperfect_ai.analysis.static_analyzer import (
    calculate_v_taper_ratio, 
    get_v_taper_score,
    analyze_muscularity,
    analyze_conditioning,
    calculate_total_package_score
)
from poseperfect_ai.analysis.dynamic_analyzer import (
    deconstruct_routine,
    analyze_stability,
    analyze_stage_presence,
    analyze_flow
)

# --- Page Configuration ---
st.set_page_config(
    page_title="PosePerfect AI",
    page_icon="💪",
    layout="wide"
)

# --- Analysis Functions ---

def run_static_analysis(image_bytes, division, pose):
    """Contains the full logic to run the static analysis and render the results."""
    with st.spinner("Performing full analysis..."):
        annotated_image, pose_landmarks = preprocess_for_static_analysis(image_bytes)
        st.success("Preprocessing Complete!")

        if pose_landmarks:
            height, width, _ = annotated_image.shape
            v_taper_ratio = calculate_v_taper_ratio(pose_landmarks, width, height)
            v_taper_score = get_v_taper_score(v_taper_ratio)
            muscularity_results = analyze_muscularity(annotated_image)
            muscularity_score = muscularity_results["Overall Fullness"]
            conditioning_results = analyze_conditioning(annotated_image)
            conditioning_score = conditioning_results["Overall Conditioning"]
            total_score = calculate_total_package_score(v_taper_score, muscularity_score, conditioning_score)

            st.header("Judge's Report Card")
            tab1, tab2, tab3 = st.tabs(["🏆 Dashboard", "📊 Diagnostics", "🧠 Coaching"])

            with tab1:
                st.subheader(f"Total Package Score: {total_score}")
                st.progress(total_score)
                st.info("This score represents the complete package, balancing symmetry, muscularity, and conditioning.")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(label="Symmetry (V-Taper)", value=v_taper_score)
                with col2:
                    st.metric(label="Muscularity", value=muscularity_score, help="Placeholder score")
                with col3:
                    st.metric(label="Conditioning", value=conditioning_score, help="Placeholder score")

            with tab2:
                st.subheader("Diagnostic Breakdown")
                img_col, data_col = st.columns([2,3])
                with img_col:
                    st.image(annotated_image, caption="Preprocessed image with landmarks.", use_column_width=True)
                with data_col:
                    st.write("**V-Taper Details:**")
                    st.text(f"Shoulder-to-Waist Ratio: {v_taper_ratio:.2f}")
                    st.write("--- ")
                    st.write("**Muscularity Details (Placeholder):**")
                    st.json(muscularity_results)
                    st.write("--- ")
                    st.write("**Conditioning Details (Placeholder):**")
                    st.json(conditioning_results)

            with tab3:
                st.subheader("AI-Generated Coaching")
                st.write("Based on your scores, here are the key areas to focus on:")
                if v_taper_score < 80:
                    st.warning(f"**V-Taper (Score: {v_taper_score}):** Your shoulder-to-waist ratio is the primary area for improvement.")
                else:
                    st.success(f"**V-Taper (Score: {v_taper_score}):** Your V-Taper is a dominant strong point!")
                st.info("More detailed coaching will be available when the Anatomist Module is fully trained.")
        else:
            st.error("Could not detect a pose in the image. Please try a different photo.")

# --- UI Rendering ---
st.title("PosePerfect AI 💪")
st.write("The definitive training partner for competitive bodybuilders.")

# --- Sidebar ---
with st.sidebar:
    st.header("Setup")
    analysis_mode = st.radio("Select Analyzer Mode", ["Static (Image)", "Dynamic (Video)"], key="analysis_mode")
    
    division = st.selectbox(
        "Select Your Division:",
        ("Men's Physique", "Classic Physique", "Men's Bodybuilding", "Bikini", "Wellness", "Figure", "Women's Physique")
    )
    
    pose = None
    if analysis_mode == "Static (Image)":
        if division == "Men's Physique":
            pose = st.selectbox("Select Your Pose:", ("Front Pose", "Back Pose"))
        else:
            pose = st.selectbox("Select Your Pose:", ("Pose options not yet available for this division.",))

# --- Main Panel ---
st.header(f"{analysis_mode} Dashboard")

if analysis_mode == "Static (Image)":
    uploaded_file = st.file_uploader("Upload your image (JPG, PNG)", type=["jpg", "png"])
    if uploaded_file:
        if st.button("Analyze Pose"):
            run_static_analysis(uploaded_file.getvalue(), division, pose)
else: # Dynamic (Video) Mode
    uploaded_file = st.file_uploader("Upload your video (MP4, MOV, AVI)", type=["mp4", "mov", "avi"])
    if uploaded_file:
        st.video(uploaded_file)
        if st.button("Analyze Routine"):
            with st.spinner("Deconstructing routine... This may take a moment."):
                video_bytes = uploaded_file.getvalue()
                routine_timeline = deconstruct_routine(video_bytes)
                
                st.success("Routine deconstruction complete!")
                st.header("Detected Routine Timeline")
                st.info("This is a placeholder timeline. The real version will identify poses from your video.")

                for i, phase in enumerate(routine_timeline):
                    if phase['type'] == 'Held Pose':
                        with st.expander(f"✅ Phase {i+1}: {phase['details']} ({phase['start_time']:.1f}s - {phase['end_time']:.1f}s)"):
                            st.write(f"**Duration:** {phase['end_time'] - phase['start_time']:.1f} seconds")
                            
                            # Call placeholder analysis functions for poise and presence
                            stability_score = analyze_stability(None, None)
                            presence_scores = analyze_stage_presence(None)

                            st.write("**Pose & Poise Critique (Placeholders):**")
                            scol1, scol2 = st.columns(2)
                            with scol1:
                                st.metric("Stability Score", f"{stability_score} / 100")
                            with scol2:
                                st.metric("Stage Presence", f"{presence_scores['overall_presence_score']} / 100")

                            if st.button("Run Full Static Analysis on this Pose", key=f"analyze_{i}"):
                                st.warning("Feature not yet implemented. This will run the full static analysis on a frame from this pose.")
                    else: # Transition
                        with st.expander(f"🔄 Phase {i+1}: {phase['details']} ({phase['start_time']:.1f}s - {phase['end_time']:.1f}s)"):
                            st.write(f"**Duration:** {phase['end_time'] - phase['start_time']:.1f} seconds")
                            
                            # Call placeholder analysis function for flow
                            flow_score = analyze_flow(None, None)
                            st.metric("Flow Score", f"{flow_score} / 100", help="Measures the smoothness of the transition.")