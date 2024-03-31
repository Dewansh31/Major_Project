import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image, ImageDraw
import imagerec
from googletrans import Translator
from languages import *

st.set_page_config(
    layout="wide",
    initial_sidebar_state='expanded',
)

# sidebar
with st.sidebar:
    selected = option_menu('NeuroVision', [
        'Home',
        'Brain Tumor Predictor',
        'About Us',
    ],
                           # icons=['', 'person'],
                           default_index=0)

    target_language = st.selectbox(
        'Select Language',
        languages)

    translate = st.button('Change Language')

if selected == 'Home':
    title_text = st.title("NeuroVision")

    # st.title("Language Translation App")
    # source_text = st.text_area("Enter text to translate:")
    # target_language = st.selectbox("Select target language:", languages)
    # translate = st.button('Translate')
    # if translate:
    #     translator = Translator()
    #     out = translator.translate(title_text.text, dest=target_language)
    #     st.write(out.text)


    st.write('<style>div.row-widget.stMarkdown { font-size: 24px; }</style>', unsafe_allow_html=True)

    st.markdown(""" <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style> """, unsafe_allow_html=True)


    def page_layout():

        session_state = st.session_state
        session_state.desc_var = "NeuroVision introduces a cutting-edge approach to brain tumor detection utilizing MRI images. This system amalgamates sophisticated machine learning algorithms, particularly Convolutional Neural Networks (CNNs), to analyze MRI scans swiftly and accurately. Similar to HealthVision, our platform is engineered to pinpoint ailments, but with a specialized focus on neurological conditions like brain tumors. By harnessing advanced computational techniques, NeuroVision aims to revolutionize medical diagnostics, offering precise and timely identification of brain tumors through the interpretation of MRI data."

        desc = st.write(session_state.desc_var)

        if translate:
            translator = Translator()
            out = translator.translate(session_state.desc_var, dest=target_language)
            session_state.desc_var = out.text

        st.markdown("## Benefits:")
        st.write("- Fast and accurate diagnosis of tumors")
        st.write("- Non-invasive and painless diagnosis using MRI")
        st.write("- Accessible from anywhere, anytime")
        st.write("- Easy to use and user friendly Interface")

        st.markdown("## Relevance:")
        st.write("- NeuroVision can diagnose various tumors, like Glioma, Meningioma, Pituitary")
        st.write("- The app can be used by doctors, hospitals, and patients")
        st.write("- HealthVision can improve the accuracy and speed of disease diagnosis")

        st.markdown("## Uses:")
        st.write("- Hospitals and clinics can use HealthVision to diagnose diseases more quickly")
        st.write(
            "- Patients can use NeuroVision to get a quick and accurate diagnosis without the need for invasive procedures")
        st.write("- NeuroVision can be used to screen large populations for brain tumors")

        st.markdown("## Future Scope:")
        st.write(
            " - Enhanced Accuracy: Continued advancements in machine learning and artificial intelligence algorithms will improve the accuracy and reliability of brain tumor detection systems. Integration of deep learning techniques and access to larger datasets will enable more precise identification and characterization of tumors.")
        st.write(
            " - Real-time Monitoring: Development of real-time monitoring capabilities will allow continuous assessment of tumor growth and response to treatment. This can enable timely intervention and adjustments in treatment strategies, improving patient care and prognosis.")
        st.write(
            " - Integration with Healthcare Ecosystem: Brain tumor detection systems will likely become seamlessly integrated into the broader healthcare ecosystem. This includes interoperability with electronic health records, communication with other medical devices, and integration into telemedicine platforms for remote consultations and monitoring.")


    # Render page layout
    page_layout()

if selected == 'Brain Tumor Predictor':

    uploaded_file = None

    st.title("Brain Tumor Predictor")

    st.write('<style>div.row-widget.stMarkdown { font-size: 24px; }</style>', unsafe_allow_html=True)

    st.write("""There are several types of brain tumors, including:

    Glioma: A type of tumor that originates in the glial cells, which are the supportive cells in the brain. Gliomas can be either low-grade (slow-growing) or high-grade (fast-growing) and can affect different parts of the brain.

    Meningioma: A tumor that arises from the meninges, which are the protective membranes that surround the brain and spinal cord. Meningiomas are usually benign and slow-growing, and may not require treatment if they are not causing symptoms.

    Pituitary adenoma: A tumor that develops in the pituitary gland, which is located at the base of the brain. Pituitary adenomas can affect hormone production and cause a variety of symptoms, depending on the hormones that are affected.""")
    st.divider()
    st.write(
        "Hence, we have developed A Convolutional Neural Network (CNN) to predict whether the MRI Scan of the brain has a tumour or not. It has been trained on more than 1000 images divided into four classes, to upto 50 epochs.")
    st.divider()
    uploaded_file = st.file_uploader("Choose a File", type=['jpg', 'png', 'jpeg'])

    if uploaded_file != None:
        st.image(uploaded_file)
    x = st.button("Predict")
    if x:
        with st.spinner("Predicting..."):
            y, conf = imagerec.imagerecognise(uploaded_file, "BrainTumuorModel.h5",
                                              "BrainTumuorLabels.txt")
        if y.strip() == "Safe":
            st.markdown("Safe")
            # components.html(
            #     """
            #     <style>
            #     h1{
            #
            #         background: -webkit-linear-gradient(0.25turn,#01CCF7, #8BF5F5);
            #         -webkit-background-clip: text;
            #         -webkit-text-fill-color: transparent;
            #         font-family: "Source Sans Pro", sans-serif;
            #     }
            #     </style>
            #     <h1>It is Negative for Brain Tumors</h1>
            #     """
            # )
        elif y.strip() == "Glioma":
            st.markdown("Glioma")
            # components.html(
            #     """
            #     <style>
            #     h1{
            #         background: -webkit-linear-gradient(0.25turn,#FF4C4B, #F70000);
            #         -webkit-background-clip: text;
            #         -webkit-text-fill-color: transparent;
            #         font-family: "Source Sans Pro", sans-serif;
            #     }
            #     </style>
            #     <h1>It is Positive for Glioma</h1>
            #     """
            # )
        elif y.strip() == "Meningioma":
            st.markdown("Meningioma")
            # components.html(
            #     """
            #     <style>
            #     h1{
            #         background: -webkit-linear-gradient(0.25turn,#FF4C4B, #F70000);
            #         -webkit-background-clip: text;
            #         -webkit-text-fill-color: transparent;
            #         font-family: "Source Sans Pro", sans-serif;
            #     }
            #     </style>
            #     <h1>It is Positive for Meningioma</h1>
            #     """
            # )
        else:
            st.markdown(" Positive for Pituitary")
            # components.html(
            #     """
            #     <style>
            #     h1{
            #         background: -webkit-linear-gradient(0.25turn,#FF4C4B, #F70000);
            #         -webkit-background-clip: text;
            #         -webkit-text-fill-color: transparent;
            #         font-family: "Source Sans Pro", sans-serif;
            #     }
            #     </style>
            #     <h1>It is Positive for Pituitary</h1>
            #     """
            # )

if selected == 'About Us':
    st.title("About Us")
    st.write("Welcome to our final year Computer Science Engineering project!")
    st.write("This project is a culmination of our efforts and dedication.")

    st.header("Meet the Team")

    col1, col2, = st.columns((1, 2))

    with col1:
        st.image("DY.jpg", width=200)

    with col2:
        st.subheader("Dewansh Yadav")
        st.write("Dewansh is a final year Computer Science Engineering student.")
        st.write("His interests include machine learning, web development, data science and artificial intelligence.")
        st.write("Contact: dewanshyadaw8@gmail.com , +91 7879690005")

    col3, col4, = st.columns((1, 2))

    with col3:
        st.image("RK.jpeg", width=200)

    with col4:
        st.subheader("Rakshit Yadav")
        st.write("Rakshit is also a final year Computer Science Engineering student.")
        st.write("His interests include data science, software development, and artificial intelligence.")
        st.write("Contact: yadavrakshit097@gmail.com , +91 7879426699")
