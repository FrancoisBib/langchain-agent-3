# ref: https://github.com/twy80/LangChain_llm_Agent/tree/main
import streamlit as st
import os, base64, re, requests, datetime, time, json
import matplotlib.pyplot as plt
from io import BytesIO
from functools import partial
from tempfile import NamedTemporaryFile
from audio_recorder_streamlit import audio_recorder
from PIL import Image, UnidentifiedImageError
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_community import GoogleSearchAPIWrapper
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage, AIMessage
from langchain_community.utilities import BingSearchAPIWrapper
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.tools import Tool, tool
from langchain.tools.retriever import create_retriever_tool
# from langchain.agents import create_openai_tools_agent
from langchain.agents import create_tool_calling_agent
from langchain.agents import create_react_agent
from langchain.agents import AgentExecutor
from langchain_community.agent_toolkits.load_tools import load_tools
# from langchain_experimental.tools import PythonREPLTool
from langchain_experimental.utilities import PythonREPL
from langchain.callbacks.base import BaseCallbackHandler
from pydantic import BaseModel, Field
# The following are for type annotations
from typing import Union, List, Literal, Optional, Dict, Any, Annotated
from matplotlib.figure import Figure
from streamlit.runtime.uploaded_file_manager import UploadedFile
from openai._legacy_response import HttpxBinaryResponseContent
from tempfile import NamedTemporaryFile, TemporaryDirectory

# Load API keys 
try:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    os.environ["BING_SUBSCRIPTION_KEY"] = st.secrets.get("BING_SUBSCRIPTION_KEY", "")
    os.environ["GOOGLE_API_KEY"] = st.secrets.get("GOOGLE_API_KEY", "")
    os.environ["GOOGLE_CSE_ID"] = st.secrets.get("GOOGLE_CSE_ID", "")
except KeyError as e:
    st.error(f"Missing required secret: {e}. Please set it in Hugging Face Space secrets.")
    st.stop()

def initialize_session_state_variables() -> None:
    """
    Initialize all the session state variables.
    """
    default_values = {
        "ready": False,
        "openai": None,
        "history": [],
        "model_type": "GPT Models from OpenAI",
        "agent_type": 2 * ["Tool Calling"],
        "ai_role": 2 * ["You are a helpful AI assistant."],
        "prompt_exists": False,
        "temperature": [0.7, 0.7],
        "audio_bytes": None,
        "mic_used": False,
        "audio_response": None,
        "image_url": None,
        "image_description": None,
        "uploader_key": 0,
        "tool_names": [[], []],
        "bing_subscription_validity": False,
        "google_cse_id_validity": False,
        "vector_store_message": None,
        "retriever_tool": None,
        "show_uploader": False
    }

    for key, value in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = value



class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: Any, **kwargs) -> None:
        new_text = self._extract_text(token)
        if new_text:
            self.text += new_text
            self.container.markdown(self.text)

    def _extract_text(self, token: Any) -> str:
        if isinstance(token, str):
            return token
        elif isinstance(token, list):
            return ''.join(self._extract_text(t) for t in token)
        elif isinstance(token, dict):
            return token.get('text', '')
        else:
            return str(token)


def check_api_keys() -> None:
    # Unset this flag to check the validity of the OpenAI API key
    st.session_state.ready = False


def message_history_to_string(extra_space: bool=True) -> str:
    """
    Return a string of the chat history contained in
    st.session_state.history.
    """

    history_list = []
    for msg in st.session_state.history:
        if isinstance(msg, HumanMessage):
            history_list.append(f"Human: {msg.content}")
        else:
            history_list.append(f"AI: {msg.content}")
    new_lines = "\n\n" if extra_space else "\n"

    return new_lines.join(history_list)


def get_chat_model(
    model: str,
    temperature: float,
    callbacks: List[BaseCallbackHandler]
) -> Union[ChatOpenAI, ChatAnthropic, ChatGoogleGenerativeAI, None]:

    """
    Get the appropriate chat model based on the given model name.
    """

    model_map = {
        "gpt-": ChatOpenAI,
    }
    for prefix, ModelClass in model_map.items():
        if model.startswith(prefix):
            return ModelClass(
                model=model,
                temperature=temperature,
                streaming=True,
                callbacks=callbacks
            )
    return None


def process_with_images(
    llm: Union[ChatOpenAI, ChatAnthropic, ChatGoogleGenerativeAI],
    message_content: str,
    image_urls: List[str]
) -> str:

    """
    Process the given history query with associated images using a language model.
    """

    content_with_images = (
        [{"type": "text", "text": message_content}] +
        [{"type": "image_url", "image_url": {"url": url}} for url in image_urls]
    )
    message_with_images = [HumanMessage(content=content_with_images)]

    return llm.invoke(message_with_images).content


def process_with_tools(
    llm: Union[ChatOpenAI, ChatAnthropic, ChatGoogleGenerativeAI],
    tools: List[Tool],
    agent_type: str,
    agent_prompt: str,
    history_query: dict
) -> str:

    """
    Create an AI agent based on the specified agent type and tools,
    then use this agent to process the given history query.
    """

    if agent_type == "Tool Calling":
        agent = create_tool_calling_agent(llm, tools, agent_prompt)
    else:
        agent = create_react_agent(llm, tools, agent_prompt)

    agent_executor = AgentExecutor(
        agent=agent, tools=tools, max_iterations=5, verbose=False,
        handle_parsing_errors=True,
    )

    return agent_executor.invoke(history_query)["output"]


def run_agent(
    query: str,
    model: str,
    tools: List[Tool],
    image_urls: List[str],
    temperature: float=0.7,
    agent_type: Literal["Tool Calling", "ReAct"]="Tool Calling",
) -> Union[str, None]:
    """
    Generate text based on user queries.
    Args:
        query: User's query
        model: LLM like "gpt-4o"
        tools: list of tools such as Search and Retrieval
        image_urls: List of URLs for images
        temperature: Value between 0 and 1. Defaults to 0.7
        agent_type: 'Tool Calling' or 'ReAct'
    Return:
        generated text
    """

    try:
        # Ensure retriever tool is included when "Retrieval" is selected
        if "Retrieval" in st.session_state.tool_names[0]:
            if st.session_state.retriever_tool:
                retriever_tool_name = "retriever"  # Ensure naming consistency
                if retriever_tool_name not in [tool.name for tool in tools]:
                    tools.append(st.session_state.retriever_tool)
                    st.write(f"✅ **{retriever_tool_name} tool has been added successfully.**")
            else:
                st.error("❌ Retriever tool is not initialized. Please create a vector store first.")
                return None  # Exit early to avoid broken tool usage

        # Debugging: Print final tools list
        st.write("**Final Tools Being Used:**", [tool.name for tool in tools])

        if "retriever" in [tool.name for tool in tools]:
            st.success("✅ Retriever tool is confirmed and ready for use.")
        elif "Retrieval" in st.session_state.tool_names[0]:
            st.warning("⚠️ 'Retrieval' was selected but the retriever tool is missing!")

        # Initialize the LLM model
        llm = get_chat_model(model, temperature, [StreamHandler(st.empty())])
        if llm is None:
            st.error(f"❌ Unsupported model: {model}", icon="🚨")
            return None
        
        # Prepare chat history
        if agent_type == "Tool Calling":
            chat_history = st.session_state.history
        else:
            chat_history = message_history_to_string()

        history_query = {"chat_history": chat_history, "input": query}

        # Generate message content
        message_with_no_image = st.session_state.chat_prompt.invoke(history_query)
        message_content = message_with_no_image.messages[0].content

        if image_urls:
            # Handle images if provided
            generated_text = process_with_images(llm, message_content, image_urls)
            human_message = HumanMessage(
                content=query, additional_kwargs={"image_urls": image_urls}
            )
        elif tools:
            # Use tools for query execution
            generated_text = process_with_tools(
                llm, tools, agent_type, st.session_state.agent_prompt, history_query
            )
            human_message = HumanMessage(content=query)
        else:
            # Fall back to basic query execution without tools
            generated_text = llm.invoke(message_with_no_image).content
            human_message = HumanMessage(content=query)

        # Convert response into plain text
        if isinstance(generated_text, list):
            generated_text = generated_text[0]["text"]

        # Update conversation history
        st.session_state.history.append(human_message)
        st.session_state.history.append(AIMessage(content=generated_text))

        return generated_text

    except Exception as e:
        st.error(f"An error occurred: {e}", icon="🚨")
        return None


def openai_create_image(
    description: str, model: str="dall-e-3", size: str="1024x1024"
) -> Optional[str]:

    """
    Generate image based on user description.

    Args:
        description: User description
        model: Default set to "dall-e-3"
        size: Pixel size of the generated image

    Return:
        URL of the generated image
    """

    try:
        with st.spinner("AI is generating..."):
            response = st.session_state.openai.images.generate(
                model=model,
                prompt=description,
                size=size,
                quality="standard",
                n=1,
            )
        image_url = response.data[0].url
    except Exception as e:
        image_url = None
        st.error(f"An error occurred: {e}", icon="🚨")

    return image_url


def get_vector_store(uploaded_files: List[UploadedFile]) -> Optional[FAISS]:
    """
    Take a list of UploadedFile objects as input, and return a FAISS vector store.
    """
    if not uploaded_files:
        return None

    documents = []
    loader_map = {
        ".pdf": PyPDFLoader,
        ".txt": TextLoader,
        ".docx": Docx2txtLoader
    }

    try:
        # Use a temporary directory instead of a fixed 'files/' directory
        with TemporaryDirectory() as temp_dir:
            for uploaded_file in uploaded_files:
                # Create a temporary file in the system's temporary directory
                with NamedTemporaryFile(dir=temp_dir, delete=False) as temp_file:
                    temp_file.write(uploaded_file.getbuffer())
                    filepath = temp_file.name
                
                file_ext = os.path.splitext(uploaded_file.name.lower())[1]
                loader_class = loader_map.get(file_ext)
                if not loader_class:
                    st.error(f"Unsupported file type: {file_ext}", icon="🚨")
                    return None

                # Load the document using the selected loader
                loader = loader_class(filepath)
                documents.extend(loader.load())

        with st.spinner("Vector store in preparation..."):
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200
            )
            doc = text_splitter.split_documents(documents)

            # Choose embeddings
            if st.session_state.model_type == "GPT Models from OpenAI":
                embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=1536)
            else:
                embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

            # Create FAISS vector database
            vector_store = FAISS.from_documents(doc, embeddings)

    except Exception as e:
        vector_store = None
        st.error(f"An error occurred: {e}", icon="🚨")

    return vector_store



def get_retriever() -> None:
    """
    Upload document(s), create a vector store, prepare a retriever tool,
    save the tool to the variable st.session_state.retriever_tool.
    """

    # Section Title
    st.write("")
    st.write("**Query Document(s)**")

    # File Upload Input
    uploaded_files = st.file_uploader(
        label="Upload an article",
        type=["txt", "pdf", "docx"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        key="document_upload_" + str(st.session_state.uploader_key),
    )

    # Check if files are uploaded
    if uploaded_files:
        # Use a unique button key to avoid duplicate presses
        if st.button(label="Create the vector store", key=f"create_vector_{st.session_state.uploader_key}"):
            st.info("Creating the vector store and initializing the retriever tool...")

            # Attempt to create the vector store
            vector_store = get_vector_store(uploaded_files)

            if vector_store:
                uploaded_file_names = [file.name for file in uploaded_files]
                st.session_state.vector_store_message = (
                    f"Vector store for :blue[[{', '.join(uploaded_file_names)}]] is ready!"
                )

                # Initialize retriever and create tool
                retriever = vector_store.as_retriever()
                st.session_state.retriever_tool = create_retriever_tool(
                    retriever,
                    name="retriever",
                    description="Search uploaded documents for information when queried.",
                )

                # Add "Retrieval" to the tools list if not already present
                if "Retrieval" not in st.session_state.tool_names[0]:
                    st.session_state.tool_names[0].append("Retrieval")

                st.success("✅ Retriever tool has been successfully initialized and is ready to use.")

                # Debugging output
                st.write("**Current Tools:**", st.session_state.tool_names[0])
            else:
                st.error("❌ Failed to create vector store. Please check the uploaded files (supported formats: txt, pdf, docx).")
    else:
        st.info("Please upload document(s) to create the vector store.")




def display_text_with_equations(text: str):
    # Replace inline LaTeX equation delimiters \\( ... \\) with $
    modified_text = text.replace("\\(", "$").replace("\\)", "$")

    # Replace block LaTeX equation delimiters \\[ ... \\] with $$
    modified_text = modified_text.replace("\\[", "$$").replace("\\]", "$$")

    # Use st.markdown to display the formatted text with equations
    st.markdown(modified_text)


def read_audio(audio_bytes: bytes) -> Optional[str]:
    """
    Read audio bytes and return the corresponding text.
    """
    try:
        audio_data = BytesIO(audio_bytes)
        audio_data.name = "recorded_audio.wav"  # dummy name

        transcript = st.session_state.openai.audio.transcriptions.create(
            model="whisper-1", file=audio_data
        )
        text = transcript.text
    except Exception as e:
        text = None
        st.error(f"An error occurred: {e}", icon="🚨")

    return text


def input_from_mic() -> Optional[str]:
    """
    Convert audio input from mic to text and return it.
    If there is no audio input, None is returned.
    """

    time.sleep(0.5)
    audio_bytes = audio_recorder(
        pause_threshold=3.0, text="Speak", icon_size="2x",
        recording_color="#e87070", neutral_color="#6aa36f"        
    )

    if audio_bytes == st.session_state.audio_bytes or audio_bytes is None:
        return None
    else:
        st.session_state.audio_bytes = audio_bytes
        return read_audio(audio_bytes)


def perform_tts(text: str) -> Optional[HttpxBinaryResponseContent]:
    """
    Take text as input, perform text-to-speech (TTS),
    and return an audio_response.
    """

    try:
        with st.spinner("TTS in progress..."):
            audio_response = st.session_state.openai.audio.speech.create(
                model="tts-1",
                voice="fable",
                input=text,
            )
    except Exception as e:
        audio_response = None
        st.error(f"An error occurred: {e}", icon="🚨")

    return audio_response


def play_audio(audio_response: HttpxBinaryResponseContent) -> None:
    """
    Take an audio response (a bytes-like object)
    from TTS as input, and play the audio.
    """

    audio_data = audio_response.read()

    # Encode audio data to base64
    b64 = base64.b64encode(audio_data).decode("utf-8")

    # Create a markdown string to embed the audio player with the base64 source
    md = f"""
        <audio controls autoplay style="width: 100%;">
        <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        Your browser does not support the audio element.
        </audio>
        """

    # Use Streamlit to render the audio player
    st.markdown(md, unsafe_allow_html=True)


def image_to_base64(image: Image) -> str:
    """
    Convert an image object from PIL to a base64-encoded image,
    and return the resulting encoded image as a string to be used
    in place of a URL.
    """

    # Convert the image to RGB mode if necessary
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Save the image to a BytesIO object
    buffered_image = BytesIO()
    image.save(buffered_image, format="JPEG")

    # Convert BytesIO to bytes and encode to base64
    img_str = base64.b64encode(buffered_image.getvalue())

    # Convert bytes to string
    base64_image = img_str.decode("utf-8")

    return f"data:image/jpeg;base64,{base64_image}"


def shorten_image(image: Image, max_pixels: int=1024) -> Image:
    """
    Take an Image object as input, and shorten the image size
    if the image is greater than max_pixels x max_pixels.
    """

    if max(image.width, image.height) > max_pixels:
        if image.width > image.height:
            new_width, new_height = 1024, image.height * 1024 // image.width
        else:
            new_width, new_height = image.width * 1024 // image.height, 1024

        image = image.resize((new_width, new_height))

    return image


def upload_image_files_return_urls(
    type: List[str]=["jpg", "jpeg", "png", "bmp"]
) -> List[str]:

    """
    Upload image files, convert them to base64-encoded images, and
    return the list of the resulting encoded images to be used
    in place of URLs.
    """

    st.write("")
    st.write("**Query Image(s)**")
    source = st.radio(
        label="Image selection",
        options=("Uploaded", "From URL"),
        horizontal=True,
        label_visibility="collapsed",
    )
    image_urls = []

    if source == "Uploaded":
        uploaded_files = st.file_uploader(
            label="Upload images",
            type=type,
            accept_multiple_files=True,
            label_visibility="collapsed",
            key="image_upload_" + str(st.session_state.uploader_key),
        )
        if uploaded_files:
            try:
                for image_file in uploaded_files:
                    image = Image.open(image_file)
                    thumbnail = shorten_image(image, 300)
                    st.image(thumbnail)
                    image = shorten_image(image, 1024)
                    image_urls.append(image_to_base64(image))
            except UnidentifiedImageError as e:
                st.error(f"An error occurred: {e}", icon="🚨")
    else:
        image_url = st.text_input(
            label="URL of the image",
            label_visibility="collapsed",
            key="image_url_" + str(st.session_state.uploader_key),
        )
        if image_url:
            if is_url(image_url):
                st.image(image_url)
                image_urls = [image_url]
            else:
                st.error("Enter a proper URL", icon="🚨")

    return image_urls


def fig_to_base64(fig: Figure) -> str:
    """
    Convert a Figure object to a base64-encoded image, and return
    the resulting encoded image to be used in place of a URL.
    """

    with BytesIO() as buffer:
        fig.savefig(buffer, format="JPEG")
        buffer.seek(0)
        image = Image.open(buffer)

        return image_to_base64(image)


def is_url(text: str) -> bool:
    """
    Determine whether text is a URL or not.
    """

    regex = r"(http|https)://([\w_-]+(?:\.[\w_-]+)+)(:\S*)?"
    p = re.compile(regex)
    match = p.match(text)
    if match:
        return True
    else:
        return False


def reset_conversation() -> None:
    """
    Reset the session_state variables for resetting the conversation.
    """

    st.session_state.history = []
    st.session_state.ai_role[1] = st.session_state.ai_role[0]
    st.session_state.prompt_exists = False
    st.session_state.temperature[1] = st.session_state.temperature[0]
    st.session_state.audio_response = None
    st.session_state.vector_store_message = None
    st.session_state.tool_names[1] = st.session_state.tool_names[0]
    st.session_state.agent_type[1] = st.session_state.agent_type[0]
    st.session_state.retriever_tool = None
    st.session_state.uploader_key = 0


def switch_between_apps() -> None:
    """
    Keep the chat settings when switching the mode.
    """

    st.session_state.temperature[1] = st.session_state.temperature[0]
    st.session_state.ai_role[1] = st.session_state.ai_role[0]
    st.session_state.tool_names[1] = st.session_state.tool_names[0]
    st.session_state.agent_type[1] = st.session_state.agent_type[0]


@tool
def python_repl(
    code: Annotated[str, "The python code to execute to generate your chart."],
):
    """Use this to execute python code. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user."""
    try:
        result = PythonREPL().run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    result_str = f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"
    return (
        result_str + "\n\nIf you have completed all tasks, respond with FINAL ANSWER."
    )


def set_tools() -> List[Tool]:
    """
    Set and return the tools for the agent. Tools that can be selected
    are internet_search, arxiv, wikipedia, python_repl, and retrieval.
    A Bing Subscription Key or Google CSE ID is required for internet_search.
    """

    class MySearchToolInput(BaseModel):
        query: str = Field(description="search query to look up")

    # Load tools
    arxiv = load_tools(["arxiv"])[0]
    wikipedia = load_tools(["wikipedia"])[0]
    # Python REPL is directly used here
    tool_dictionary = {
        "ArXiv": arxiv,
        "Wikipedia": wikipedia,
        "Python_REPL": python_repl,
        "Retrieval": st.session_state.retriever_tool if st.session_state.retriever_tool else None
    }
    tool_options = ["ArXiv", "Wikipedia", "Python_REPL", "Retrieval"]

    # Add Search tool dynamically if credentials are valid
    if st.session_state.bing_subscription_validity:
        search = BingSearchAPIWrapper()
    elif st.session_state.google_cse_id_validity:
        search = GoogleSearchAPIWrapper()
    else:
        search = None

    if search is not None:
        internet_search = Tool(
            name="internet_search",
            description=(
                "A search engine for comprehensive, accurate, and trusted results. "
                "Useful for when you need to answer questions about current events. "
                "Input should be a search query."
            ),
            func=partial(search.results, num_results=5),
            args_schema=MySearchToolInput,
        )
        tool_options.insert(0, "Search")
        tool_dictionary["Search"] = internet_search

    # UI for selecting tools
    st.write("")
    st.write("**Tools**")
    tool_names = st.multiselect(
        label="assistant tools",
        options=tool_options,
        default=st.session_state.tool_names[1],
        label_visibility="collapsed",
    )

    # Instructions if Search tool is unavailable
    if "Search" not in tool_options:
        st.write(
            "<small>Tools are disabled when images are uploaded and queried. "
            "To search the internet, obtain your Bing Subscription Key "
            "[here](https://portal.azure.com/) or Google CSE ID "
            "[here](https://programmablesearchengine.google.com/about/), "
            "and enter it in the sidebar. Once entered, 'Search' will be displayed "
            "in the list of tools. Note also that PythonREPL from LangChain is still "
            "in the experimental phase, so caution is advised.</small>",
            unsafe_allow_html=True,
        )
    else:
        st.write(
            "<small>Tools are disabled when images are uploaded and queried. "
            "Note also that PythonREPL from LangChain is still in the experimental phase, "
            "so caution is advised.</small>",
            unsafe_allow_html=True,
        )

    # Handle Retrieval tool initialization
    if "Retrieval" in tool_names:
        if not st.session_state.retriever_tool:
            st.info("Creating the vector store and initializing the retriever tool...")
            get_retriever()
        if st.session_state.retriever_tool:
            st.success("Retriever tool is ready for querying.")
            tool_dictionary["Retrieval"] = st.session_state.retriever_tool
        else:
            st.error("Failed to initialize the retriever tool. Please upload the document again.")
            tool_names.remove("Retrieval")  # Prevent broken Retrieval tool

    # Final tool selection
    tools = [
        tool_dictionary[key]
        for key in tool_names if tool_dictionary[key] is not None
    ]

    st.write("**Tools selected in set_tools:**", [tool.name for tool in tools])
    st.session_state.tool_names[0] = tool_names

    return tools



def set_prompts(agent_type: Literal["Tool Calling", "ReAct"]) -> None:
    """
    Set chat and agent prompts for two different types of agents:
    Tool Calling and ReAct.
    """

    if agent_type == "Tool Calling":
        st.session_state.chat_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                f"{st.session_state.ai_role[0]} Your goal is to provide "
                "answers to human inquiries. Should the information not "
                "be available, inform the human explicitly that "
                "the answer could not be found."
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])
        st.session_state.agent_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                f"{st.session_state.ai_role[0]} Your goal is to provide answers to human inquiries. "
                "You should specify the source of your answers, whether they are based on internet search "
                "results ('internet_search'), scientific articles from arxiv.org ('arxiv'), Wikipedia documents ('wikipedia'), "
                "uploaded documents ('retriever'), or your general knowledge. "
                "Use the 'retriever' tool to answer questions specifically related to uploaded documents. "
                "If you cannot find relevant information in the documents using the 'retriever' tool, explicitly inform the user. "
                "Use Markdown syntax and include relevant sources, such as links (URLs)."
            ),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
    else:
        st.session_state.chat_prompt = ChatPromptTemplate.from_template(
            f"{st.session_state.ai_role[0]} "
            "Your goal is to provide answers to human inquiries. "
            "Should the information not be available, inform the human "
            "explicitly that the answer could not be found.\n\n"
            "{chat_history}\n\nHuman: {input}\n\n"
            "AI: "
        )
        st.session_state.agent_prompt = ChatPromptTemplate.from_template(
            f"{st.session_state.ai_role[0]} "
            "Your goal is to provide answers to human inquiries. "
            "When giving your answers, tell the human what your response "
            "is based on and which tools you use. Use Markdown syntax "
            "and include relevant sources, such as links (URLs), following "
            "MLA format. Should the information not be available, inform "
            "the human explicitly that the answer could not be found.\n\n"
            "TOOLS:\n"
            "------\n\n"
            "You have access to the following tools:\n\n"
            "{tools}\n\n"
            "To use a tool, please use the following format:\n\n"
            "Thought: Do I need to use a tool? Yes\n"
            "Action: the action to take, should be one of [{tool_names}]\n"
            "Action Input: the input to the action\n"
            "Observation: the result of the action\n\n"
            "When you have a response to say to the Human, "
            "or if you do not need to use a tool, you MUST use "
            "the format:\n\n"
            "Thought: Do I need to use a tool? No\n"
            "Final Answer: [your response here]\n\n"
            "Begin!\n\n"
            "Previous conversation history:\n\n"
            "{chat_history}\n\n"
            "New input: {input}\n"
            "{agent_scratchpad}"
        )


def print_conversation(no_of_msgs: Union[Literal["All"], int]) -> None:
    """
    Print the conversation stored in st.session_state.history.
    """

    if no_of_msgs == "All":
        no_of_msgs = len(st.session_state.history)

    for msg in st.session_state.history[-no_of_msgs:]:
        if isinstance(msg, HumanMessage):
            with st.chat_message("human"):
                st.write(msg.content)
        else:
            with st.chat_message("ai"):
                display_text_with_equations(msg.content)

        if urls := msg.additional_kwargs.get("image_urls"):
            for url in urls:
                st.image(url)

    # Play TTS
    if (
        st.session_state.model_type == "GPT Models from OpenAI"
        and st.session_state.audio_response is not None
    ):
        play_audio(st.session_state.audio_response)
        st.session_state.audio_response = None


def serialize_messages(
    messages: List[Union[HumanMessage, AIMessage]]
) -> List[Dict]:

    """
    Serialize the list of messages into a list of dicts
    """

    return [msg.dict() for msg in messages]


def deserialize_messages(
    serialized_messages: List[Dict]
) -> List[Union[HumanMessage, AIMessage]]:

    """
    Deserialize the list of messages from a list of dicts
    """

    deserialized_messages = []
    for msg in serialized_messages:
        if msg['type'] == 'human':
            deserialized_messages.append(HumanMessage(**msg))
        elif msg['type'] == 'ai':
            deserialized_messages.append(AIMessage(**msg))
    return deserialized_messages


def show_uploader() -> None:
    """
    Set the flag to show the uploader.
    """

    st.session_state.show_uploader = True


def check_conversation_keys(lst: List[Dict[str, Any]]) -> bool:
    """
    Check if all items in the given list are valid conversation entries.
    """

    return all(
        isinstance(item, dict) and
        isinstance(item.get("content"), str) and
        isinstance(item.get("type"), str) and
        isinstance(item.get("additional_kwargs"), dict)
        for item in lst
    )


def load_conversation() -> bool:
    """
    Load the conversation from a JSON file
    """

    st.write("")
    st.write("**Choose a (JSON) conversation file**")
    uploaded_file = st.file_uploader(
        label="Load conversation", type="json", label_visibility="collapsed"
    )
    if uploaded_file:
        try:
            data = json.load(uploaded_file)
            if isinstance(data, list) and check_conversation_keys(data):
                st.session_state.history = deserialize_messages(data)
                return True
            st.error(
                f"The uploaded data does not conform to the expected format.", icon="🚨"
            )
        except Exception as e:
            st.error(f"An error occurred: {e}", icon="🚨")

    return False


def create_text(model: str) -> None:
    """
    Take an LLM as input and generate text based on user input
    by calling run_agent().
    """

    # initial system prompts
    general_role = "You are a helpful AI assistant."
    english_teacher = (
        "You are an AI English teacher who analyzes texts and corrects "
        "any grammatical issues if necessary."
    )
    translator = (
        "You are an AI translator who translates English into Korean "
        "and Korean into English."
    )
    coding_adviser = (
        "You are an AI expert in coding who provides advice on "
        "good coding styles."
    )
    science_assistant = "You are an AI science assistant."
    roles = (
        general_role, english_teacher, translator,
        coding_adviser, science_assistant
    )

    with st.sidebar:
        st.write("")
        type_options = ("Tool Calling", "ReAct")
        st.write("**Agent Type**")
        st.session_state.agent_type[0] = st.sidebar.radio(
            label="Agent Type",
            options=type_options,
            index=type_options.index(st.session_state.agent_type[1]),
            label_visibility="collapsed",
        )
        agent_type = st.session_state.agent_type[0]
        if st.session_state.model_type == "GPT Models from OpenAI":
            st.write("")
            st.write("**Text to Speech**")
            st.session_state.tts = st.radio(
                label="TTS",
                options=("Enabled", "Disabled", "Auto"),
                # horizontal=True,
                index=1,
                label_visibility="collapsed",
            )
        st.write("")
        st.write("**Temperature**")
        st.session_state.temperature[0] = st.slider(
            label="Temperature (higher $\Rightarrow$ more random)",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.temperature[1],
            step=0.1,
            format="%.1f",
            label_visibility="collapsed",
        )
        st.write("")
        st.write("**Messages to Show**")
        no_of_msgs = st.radio(
            label="$\\textsf{Messages to show}$",
            options=("All", 20, 10),
            label_visibility="collapsed",
            horizontal=True,
            index=2,
        )

    st.write("")
    st.write("##### Message to AI")
    st.session_state.ai_role[0] = st.selectbox(
        label="AI's role",
        options=roles,
        index=roles.index(st.session_state.ai_role[1]),
        label_visibility="collapsed",
    )

    if st.session_state.ai_role[0] != st.session_state.ai_role[1]:
        reset_conversation()
        st.rerun()

    st.write("")
    st.write("##### Conversation with AI")

    # Print conversation
    print_conversation(no_of_msgs)

    # Reset, download, or load the conversation
    c1, c2, c3 = st.columns(3)
    c1.button(
        label="$~\:\,\,$Reset$~\:\,\,$",
        on_click=reset_conversation
    )
    c2.download_button(
        label="Download",
        data=json.dumps(serialize_messages(st.session_state.history), indent=4),
        file_name="conversation_with_agent.json",
        mime="application/json",
    )
    c3.button(
        label="$~~\:\,$Load$~~\:\,$",
        on_click=show_uploader,
    )

    if st.session_state.show_uploader and load_conversation():
        st.session_state.show_uploader = False
        st.rerun()

    # Set the agent prompts and tools
    set_prompts(agent_type)
    tools = set_tools()
    st.write("**Tools passed to run_agent:**", [tool.name for tool in tools])


    image_urls = []
    with st.sidebar:
        image_urls = upload_image_files_return_urls()

    if st.session_state.model_type == "GPT Models from OpenAI":
        audio_input = input_from_mic()
        if audio_input is not None:
            query = audio_input
            st.session_state.prompt_exists = True
            st.session_state.mic_used = True

    # Use your keyboard
    text_input = st.chat_input(placeholder="Enter your query")

    if text_input:
        query = text_input.strip()
        st.session_state.prompt_exists = True

    if st.session_state.prompt_exists:
        with st.chat_message("human"):
            st.write(query)

        with st.chat_message("ai"):
            generated_text = run_agent(
                query=query,
                model=model,
                tools=tools,
                image_urls=image_urls,
                temperature=st.session_state.temperature[0],
                agent_type=agent_type,
            )
            fig = plt.gcf()
            if fig and fig.get_axes():
                generated_image_url = fig_to_base64(fig)
                st.session_state.history[-1].additional_kwargs["image_urls"] = [
                    generated_image_url
                ]
        if (
            st.session_state.model_type == "GPT Models from OpenAI"
            and generated_text is not None
        ):
            # TTS under two conditions
            cond1 = st.session_state.tts == "Enabled"
            cond2 = st.session_state.tts == "Auto" and st.session_state.mic_used
            if cond1 or cond2:
                st.session_state.audio_response = perform_tts(generated_text)
            st.session_state.mic_used = False

        st.session_state.prompt_exists = False

        if generated_text is not None:
            st.session_state.uploader_key += 1
            st.rerun()


def create_image(model: str) -> None:
    """
    Generate image based on user description by calling openai_create_image().
    """

    # Set the image size
    with st.sidebar:
        st.write("")
        st.write("**Pixel size**")
        image_size = st.radio(
            label="$\\hspace{0.1em}\\texttt{Pixel size}$",
            options=("1024x1024", "1792x1024", "1024x1792"),
            # horizontal=True,
            index=0,
            label_visibility="collapsed",
        )

    st.write("")
    st.write("##### Description for your image")

    if st.session_state.image_url is not None:
        st.info(st.session_state.image_description)
        st.image(image=st.session_state.image_url, use_column_width=True)
    
    # Get an image description using the microphone
    if st.session_state.model_type == "GPT Models from OpenAI":
        audio_input = input_from_mic()
        if audio_input is not None:
            st.session_state.image_description = audio_input
            st.session_state.prompt_exists = True

    # Get an image description using the keyboard
    text_input = st.chat_input(
        placeholder="Enter a description for your image",
    )
    if text_input:
        st.session_state.image_description = text_input.strip()
        st.session_state.prompt_exists = True

    if st.session_state.prompt_exists:
        st.session_state.image_url = openai_create_image(
            st.session_state.image_description, model, image_size
        )
        st.session_state.prompt_exists = False
        if st.session_state.image_url is not None:
            st.rerun()


def create_text_image() -> None:
    """
    Generate text or image by using LLM models like 'gpt-4o'.
    """

    page_title = "LangChain LLM Agent"
    page_icon = "📚"

    st.set_page_config(
        page_title=page_title,
        page_icon=page_icon,
        layout="centered"
    )

    st.write(f"## {page_icon} $\,${page_title}")

    # Initialize all the session state variables
    initialize_session_state_variables()

    # Define model options directly here
    model_options = ["gpt-4o-mini", "gpt-4o", "dall-e-3"]

    # Sidebar content
    with st.sidebar:
        st.write("**Select a Model**")
        model = st.radio(
            label="Models",
            options=model_options,
            index=1,  # Default to the second option
            label_visibility="collapsed",
            on_change=switch_between_apps,
        )

        st.write("---")
        st.write("xyz", unsafe_allow_html=True)

    # Main logic for generating text or image
    if model == "dall-e-3":
        create_image(model)
    else:
        create_text(model)

if __name__ == "__main__":
    create_text_image()
