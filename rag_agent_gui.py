"""
**********************************************************************************************************************************************************
***********************************************************      RAG AGENT GUI      **********************************************************************
**********************************************************************************************************************************************************

This code is a Gradio app that allows users to interact with a language model (LLM) and a vector database (VDB) for document retrieval and question answering. 
The app supports file uploads, RAG (Retrieval-Augmented Generation), and displays both the context retrieved from the VDB and the content of uploaded files.
"""

# import for the LLM
from langchain_ollama.llms import OllamaLLM

# import for the vector database wrapper
from chroma_db_wrapper.db_wrapper import VectorDB

# import to extract text from files
from chroma_db_wrapper.file_ingestor import process_file_for_vdb

# imports for gradio  (___Message is used for the chat history)
import gradio as gr
from gradio_themes import default_theme, theme1, theme2, theme3, theme4, theme5
from langchain.schema import AIMessage, HumanMessage, SystemMessage


# command: ollama pull model_name
MODELS = ["llama3.2", "gemma3:1b"]

AUTHENTICATION_CREDENTIALS = [("kevin", "password"), ("admin", "admin")]

SYSTEM_MESSAGE = "You are a helpful assistant."


# Create an instance of the VectorDB class, loading the database from the default location
vector_db = VectorDB(load_db=True)

# Handle for the database logger
logger = vector_db.logger

# Default number of documents to retrieve
K = 5                           

# Content of the uploaded files
UPLOADED_FILES_CONTENT = ""

# Content retrieved from the vector database
RETRIEVED_CONTEXT_FROM_VDB = ""


# Function to manage the bot's response
def bot_response(message, history, *args):
    """
    This function handles the bot's response to user input. It processes uploaded files, retrieves context from the vector database, and generates a response using the LLM.
    args: additional arguments for selecting the LLM model, enabling RAG, setting the number of documents to retrieve (K), and selecting a collection.
    args[0]: str - Model name (default: "gemma3:1b")
    args[1]: bool - Enable RAG (default: False)
    args[2]: int - Number of documents to retrieve (default: 5)
    args[3]: str - Collection name (default: None)
    """

    log = f"Provided arguments: {args}\n"
    log += f"Model selected: {args[0]}\n"
    log += f"Use RAG: {args[1]}\n"
    log += f"K: {args[2]}\n"
    log += f"Collection name: {args[3]}\n"
    logger.info(log)

    model = OllamaLLM(model=args[0])                     
    use_rag = args[1] if isinstance(args[1], bool) else False
    K = args[2] if isinstance(args[2], int) else K
    if args[3] is not None:
        collection_name = args[3]
        vector_db.switch_collection(collection_name)

    log = f"Message: {message},\n History: {history}"
    logger.info(log)

    files_content = []

    if isinstance(message, dict) and "files" in message:
        for file_path in message["files"]:
            log = f"Uploaded a file with path: {file_path}\n"
            logger.info(log)
            # extract text from the file 
            document, metadatas = process_file_for_vdb(file_path)
            log = f"Extracted text from the file: {document}\n"
            logger.info(log)
            files_content.append(document)

    history_langchain_format = []
    history_langchain_format.append(SystemMessage(content=SYSTEM_MESSAGE))
    log = f"HISTORY LANGCHAIN FORMAT: {history_langchain_format}\n"
    logger.info(log)

    if history is not None:
        for item in history:
            role = item['role']
            content = item['content']

            if role == 'user':
                # 'content' might be a tuple if it's a file. We should handle that.
                user_message = content[0] if isinstance(content, tuple) and content else content
                history_langchain_format.append(HumanMessage(content=user_message))
                log = f"User message: {user_message}\n"
                logger.info(log)
            elif role == 'assistant':
                history_langchain_format.append(AIMessage(content=content))
                log = f"AI message: {content}\n"
                logger.info(log)
        log = f"Updated history langchain format: {history_langchain_format}\n"
        logger.info(log)



    if message is not None:
        # template prompt for the LLM, should include the file uploaded content, the user prompt and the context retrieved from the vector database
        full_prompt = ""

        # if the user uploaded a file, the prompt is built with the file content and the user prompt
        full_user_prompt = ""

        # text prompt inserted by the user
        user_prompt = ""

        # content of the uploaded files
        concat_files_content = ""

        if isinstance(message, dict):   
            if "text" in message:       # if the user also inserted a text prompt
                # If the message is a dictionary, extract the text (if a file is uploaded, the message will be a dictionary with "text" (user inserted text) and "files" keys (path to the file))
                user_prompt = message["text"]
            else:
                user_prompt = ""
            log = f"User prompt: {user_prompt}\n"
            logger.info(log)
        
            for content in files_content:
                if isinstance(content, list):   
                    content = " ".join(content)
                elif isinstance(content, str):
                    content = content
                else:
                    log = f"Unsupported content type: {type(content)}\n"
                    logger.error(log)
                    raise ValueError(log)
                
                concat_files_content += content + "\n "
                global UPLOADED_FILES_CONTENT
                UPLOADED_FILES_CONTENT = concat_files_content

        # if the user uploaded a file, the prompt is built with the file content and the user prompt
        if concat_files_content:
            full_user_prompt = concat_files_content + "\n" + user_prompt
            log = f"FULL USER PROMPT WAS BUILD (UPLOADED FILE): {full_user_prompt}\n"
            logger.info(log)
        # if the user did not upload a file, the prompt is built only with the user prompt
        else:
            full_user_prompt = user_prompt
            log = f"FULL USER PROMPT WAS BUILD (NO FILE UPLOADED): {full_user_prompt}\n"
            logger.info(log)

        # similarity search in the vector database (based only on the user prompt for performance reasons - no documents uploaded are used for similarity search)
        if use_rag:
            global RETRIEVED_CONTEXT_FROM_VDB
            if user_prompt:
                retrieved_docs = vector_db.retrieve_documents(query_texts=[user_prompt], k=K)
                retrieved_docs_text = [doc.page_content for doc in retrieved_docs]
                retrieved_docs_text = "\n".join(retrieved_docs_text)
                log = f"Retrieved documents from the vector database: {retrieved_docs_text}\n"
                logger.info(log)
                RETRIEVED_CONTEXT_FROM_VDB = retrieved_docs_text

                full_prompt = f"""Answer the following question: {full_user_prompt}
                Here is some context: {retrieved_docs_text}
                Make sure to answer the question based on the context provided. Do not provide any additional information or opinions.
                """
        else:
            full_prompt = f"""Answer the following question: {full_user_prompt}
            Make sure to answer the question based on the context provided. Do not provide any additional information or opinions.
            """
        log = f"FULL PROMPT: {full_prompt}\n"
        logger.info(log)
        
        history_langchain_format.append(HumanMessage(content=full_prompt))
        partial_message = ""
        for response in model.stream(history_langchain_format):
            partial_message += response
            yield partial_message



def show_context():
    global RETRIEVED_CONTEXT_FROM_VDB
    return RETRIEVED_CONTEXT_FROM_VDB

def clear_context():
    return ""

def show_file_content():
    global UPLOADED_FILES_CONTENT
    return UPLOADED_FILES_CONTENT

def clear_file_content():
    return ""


# list of checkboxes to manage RAG agent
rag_checkboxes = []

# Manage the model selection via GUI
model_dropdown = gr.Dropdown(choices=MODELS, label="Select a model", value="gemma3:1b", type="value", info="Select the model to use for the chatbot")
rag_checkboxes.append(model_dropdown)

# checkbox to enable/disable RAG
enable_rag_checkbox = gr.Checkbox(label="Enable RAG", value=False, info="Enable RAG to retrieve context from the vector database")
rag_checkboxes.append(enable_rag_checkbox)

# slider to set the number of documents to retrieve
k_slider = gr.Slider(minimum=1, maximum=10, value=5, label="K", info="Number of documents to retrieve", step=1)
rag_checkboxes.append(k_slider)

# radio button to select the collection
colletion_names = vector_db.get_collection_names()
collection_radio = gr.Radio(choices=colletion_names, label="Select a collection", value=None, type="value")
rag_checkboxes.append(collection_radio)

chat_iface = gr.ChatInterface(
    bot_response,
    multimodal=True,
    type="messages",
    textbox = gr.MultimodalTextbox(placeholder="Message the LLM...", container=False, scale=10, file_types=[".csv", ".txt", ".pdf"], file_count="multiple", sources=["upload"]),
    additional_inputs_accordion =gr.Accordion(label="Choose a model, Enable RAG, choose the parameter K and the desired collection to query", open=True),
    additional_inputs=rag_checkboxes,
    show_progress="full",
    editable=True,
    fill_width=True,

)


with gr.Blocks(theme=theme4) as demo:
    # add emoji to the title
    gr.Markdown(
        """
        <h1 style="text-align: center;">
            RAG Chatbot 🤖
        </h1>
        """
    )
    # add a description
    gr.Markdown(
        """
        <h2 style="text-align: center;">
            Chat with your documents using RAG
        </h2>
        """
    )
    chat = chat_iface.render()
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Chat with RAG Context")
            show_context_btn = gr.Button("Show context")
            context_box = gr.Textbox(label="RAG Context", placeholder="Context will be shown here", visible=True)
            clear_context_btn = gr.Button("Clear context")

        with gr.Column():
            gr.Markdown("### File Content")
            show_file_content_btn = gr.Button("Show file content")
            file_content = gr.Textbox(label="Extracted File Content", placeholder="File content will be shown here", visible=True)
            clear_file_content_btn = gr.Button("Clear file content")

    
    show_context_btn.click(show_context, inputs=[], outputs=context_box)
    clear_context_btn.click(clear_context, inputs=[], outputs=context_box)

    show_file_content_btn.click(show_file_content, inputs=[], outputs=file_content)
    clear_file_content_btn.click(clear_file_content, inputs=[], outputs=file_content)

demo.launch(auth=AUTHENTICATION_CREDENTIALS, share=True, debug=True)