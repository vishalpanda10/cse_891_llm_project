import os
import re
import torch
import torch.nn as nn
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoProcessor
from transformers import GenerationConfig, pipeline
from langchain import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain
from dotenv import load_dotenv
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.embeddings import OllamaEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from diffusers import DiffusionPipeline
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.indexes import VectorstoreIndexCreator
from PIL import Image
from io import BytesIO
from langchain.schema.runnable import RunnablePassthrough

from langchain.prompts import (
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
import clip

# In[35]:



def get_index(): 
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")


    loader = CSVLoader(file_path="prompts_unique.csv")    
    documents = loader.load()
    
    index_from_loader = FAISS.from_documents(documents, embeddings)
        
    return index_from_loader




def semantic_search(index, original_prompt): 
        
    relevant_prompts = index.similarity_search(original_prompt)    

    list_prompts = []
    for i in range(len(relevant_prompts)):
        list_prompts.append(relevant_prompts[i].page_content)
    
    return list_prompts


def clean_mistral_output(output_text):
    pattern = re.compile(r'\[/INST\]\n\s*(.*)', re.DOTALL)
    
    match = pattern.search(output_text)
    
    if match:
        cleaned_text = match.group(1).strip() 
        return cleaned_text
    else:
        pass





def get_gpt_response(original_caption, similar_captions): 


    
    gpt_template_str = """ Craft an image description (under 50 tokens) that captures the essence of a given theme, utilizing stylistic elements from the provided examples. Focus on depicting the scene's atmosphere, characters, and key details, incorporating mood, colors, and textures for depth. Let the examples guide the style of your narrative.

    Examples for inspiration:
    {similar_captions}
    """
    
    review_system_prompt = SystemMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["similar_captions"],
        template=gpt_template_str,
    )
    )

    review_human_prompt = HumanMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["original_caption"],
        template="{original_caption}",
    )
    )
    messages = [review_system_prompt, review_human_prompt]

    review_prompt_template = ChatPromptTemplate(
    input_variables=["similar_captions", "original_caption"],
    messages=messages,
    )
    output_parser = StrOutputParser()
    
    chat_model = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
    
    review_chain = review_prompt_template | chat_model | output_parser
    result = review_chain.invoke({"similar_captions": similar_captions, "original_caption": original_caption})
    
    return result


