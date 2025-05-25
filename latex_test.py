import streamlit as st

def main():
    st.title("LaTeX Rendering Test in Streamlit")

    st.markdown("""
    ## **Display Equation Test**

    Here is a correctly formatted LaTeX equation:

    \[
    \frac{\partial^\alpha C}{\partial t^\alpha} = \frac{1}{\Gamma(1 - \alpha)} \int_0^t (t - \tau)^{-\alpha} \frac{\partial C}{\partial \tau} d\tau
    \]

    And an inline equation: $E = mc^2$.

    """)

    st.markdown("""
    ### **Using st.latex Function**

    You can also use the `st.latex` function to render LaTeX:

    """)
    
    st.latex(r'''
        \frac{\partial^\alpha C}{\partial t^\alpha} = \frac{1}{\Gamma(1 - \alpha)} \int_0^t (t - \tau)^{-\alpha} \frac{\partial C}{\partial \tau} d\tau
    ''')

if __name__ == "__main__":
    main()
