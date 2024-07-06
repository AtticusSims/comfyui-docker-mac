from stable_cascade import web_demo
import gradio as gr


gradio_app = gr.Blocks()
with gradio_app:
    gr.HTML(
        """
    <h1 style='text-align: center'>
    Stable Cascade 🚀
    </h1>
    """)
    gr.HTML(
        """
        <h3 style='text-align: center'>
        Follow me for more!
        <a href='https://twitter.com/kadirnar_ai' target='_blank'>Twitter</a> | <a href='https://github.com/kadirnar' target='_blank'>Github</a> | <a href='https://www.linkedin.com/in/kadir-nar/' target='_blank'>Linkedin</a>  | <a href='https://www.huggingface.co/kadirnar/' target='_blank'>HuggingFace</a>
        </h3>
        """)
    with gr.Row():
        with gr.Column():
            web_demo()

gradio_app.queue()
gradio_app.launch(debug=True)