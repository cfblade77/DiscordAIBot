import nest_asyncio
import discord
import torch
from huggingface_hub import notebook_login
from llama_index.core import PromptTemplate
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.core import Settings
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
import sys
import logging
pip install - U transformers - -upgrade
!pip install discord
!pip install nest_asyncio
!pip install - q pypdf
!pip install - q python-dotenv
!pip install  llama-index

!pip install einops
!pip install accelerate
!pip install llama-index-llms-huggingface
!pip install llama-index-embeddings-fastembed
!pip install fastembed

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


documents = SimpleDirectoryReader("/content/data").load_data()

embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.embed_model = embed_model
Settings.chunk_size = 512


system_prompt = "You are a QA Bot.You have to understand the language of the user input , generate an answer from the provided data source in the same langauge as the user input. Reply I dont know the answer if you dont find answer of the user's query"


# This will wrap the default prompts that are internal to llama-index
query_wrapper_prompt = PromptTemplate("<|USER|>{query_str}<|QABot|>")


notebook_login()

llm = HuggingFaceLLM(
    context_window=8192,
    max_new_tokens=512,
    generate_kwargs={"temperature": 0.7, "do_sample": True},
    system_prompt=system_prompt,
    query_wrapper_prompt=query_wrapper_prompt,
    tokenizer_name="Telugu-LLM-Labs/Indic-gemma-2b-finetuned-sft-Navarasa-2.0",
    model_name="Telugu-LLM-Labs/Indic-gemma-2b-finetuned-sft-Navarasa-2.0",
    device_map="auto",
    # stopping_ids=[50278, 50279, 50277, 1, 0],
    tokenizer_kwargs={"max_length": 2048},
    # uncomment this if using CUDA to reduce memory usage
    model_kwargs={"torch_dtype": torch.float16}
)

Settings.llm = llm
index = VectorStoreIndex.from_documents(documents)
nest_asyncio.apply()

# Assuming you've already initialized 'index' for your query engine
# create query engine
query_engine = index.as_query_engine(similarity_top_k=2)

# querying


async def final_result(query):  # Make it async
    response = query_engine.query(query)
    return response


class MyClient(discord.Client):
    async def on_ready(self):
        print(f'Logged on as {self.user}!')

    async def on_message(self, message):
        if message.author == self.user:
            return

        # Assuming the bot should respond only to mentions
        if self.user.mentioned_in(message):
            query = message.content
            response = await final_result(query)  # Await the result
            await message.channel.send(response)


intents = discord.Intents.default()
intents.message_content = True

client = MyClient(intents=intents)
client.run(
    'enter your discord token here', log_handler=None)
