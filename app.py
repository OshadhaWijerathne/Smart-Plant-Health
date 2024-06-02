import streamlit as st
from streamlit_login_auth_ui_main.streamlit_login_auth_ui.widgets import __login__
#from chatbot import chatbot_func
from chatbot import *
from about import about_func

__login__obj = __login__(auth_token = "courier_auth_token",
                    company_name = "Shims",
                    width = 200, height = 250,
                    logout_button_name = 'Logout', hide_menu_bool = False,
                    hide_footer_bool = False,
                    lottie_url= 'https://lottie.host/8955ca60-0e63-4fe2-98da-e59a9e608c03/JbKk1tv2tf.json' )
                    #lottie_url = 'https://assets2.lottiefiles.com/packages/lf20_jcikwtux.json')

LOGGED_IN= __login__obj.build_login_ui()
username= __login__obj.get_username()
#LOGGED_IN, username = True,"Oshadha2000"

def main_page():
    st.markdown("# Main page 🎈")
    st.sidebar.markdown("# Main page 🎈")

def chatbot_function():
    chatbot_func(vectordb)   
     
def page3():
    st.markdown("# Page 3 🎉")
    st.sidebar.markdown("# Page 3 🎉")

vectordb = prepare_db("Chroma_Vector_Store")

page_names_to_funcs = {
    "Main Page": main_page,
    "Chatbot": chatbot_function,
    "About": about_func,
    "Page 3": page3,
}


if LOGGED_IN == True:

   st.markdown("Your Streamlit Application Begins here!")
   #st.markdown(st.session_state)
   st.write(username)
   selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
   page_names_to_funcs[selected_page]()