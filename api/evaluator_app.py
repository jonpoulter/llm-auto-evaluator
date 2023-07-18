"""
This is an API to support the LLM QA chain auto-evaluator. 
"""

import io
import os
from dotenv import load_dotenv
import sentry_sdk
import json
import time
import pypdf
import random
import logging
import itertools
import faiss
import pandas as pd
from typing import Dict, List
from json import JSONDecodeError
from langchain.llms import MosaicML
from langchain.llms import Anthropic
from langchain.llms import Replicate
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.chains import QAGenerationChain
from langchain.retrievers import SVMRetriever
from langchain.evaluation.qa import QAEvalChain
from langchain.retrievers import TFIDFRetriever
from sse_starlette.sse import EventSourceResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain.embeddings import LlamaCppEmbeddings
from langchain.embeddings import MosaicMLInstructorEmbeddings
from fastapi import FastAPI, File, UploadFile, Form
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from text_utils import GRADE_DOCS_PROMPT, GRADE_ANSWER_PROMPT, GRADE_DOCS_PROMPT_FAST, GRADE_ANSWER_PROMPT_FAST, GRADE_ANSWER_PROMPT_BIAS_CHECK, GRADE_ANSWER_PROMPT_OPENAI, QA_CHAIN_PROMPT, QA_CHAIN_PROMPT_LLAMA, QA_WITH_SOURCES_CHAIN_PROMPT
from langchain.llms import AzureOpenAI
from langchain.embeddings import OpenAIEmbeddings as AzureOpenAIEmbeddings
from langchain.chat_models import AzureChatOpenAI
import openai

load_dotenv()

if os.environ.get("OPENAI_API_TYPE") == "azure":
    #init Azure OpenAI
    openai.api_type = "azure"
    openai.api_version = os.getenv("AZURE_OPENAI_DEPLOYMENT_VERSION")
    openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
    openai.api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    #azure llm needs these set as environment variable - perhaps because of Langchain wrapper 
    os.environ["OPENAI_API_KEY"] = os.environ.get("AZURE_OPENAI_API_KEY")
    os.environ["OPENAI_API_VERSION"] = os.environ.get("AZURE_OPENAI_DEPLOYMENT_VERSION")
    os.environ["OPENAI_API_BASE"] = os.environ.get("AZURE_OPENAI_ENDPOINT")
    os.environ["OPENAI_API_TYPE"] = "azure"
else:
    openai.api_key = os.environ.get("OPENAI_OPENAI_API_KEY")
    os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_OPENAI_API_KEY")





#def set_env_openai():
#        os.environ.pop("OPENAI_API_TYPE") 
#    if os.environ.get("OPENAI_API_TYPE"):
#    os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_OPENAI_API_KEY")
#    if os.environ.get("OPENAI_API_BASE"):
#        os.environ.pop("OPENAI_API_BASE")
#    if os.environ.get("OPENAI_API_VERSION"):
#        os.environ.pop("OPENAI_API_VERSION")

def generate_eval(text, chunk, logger):
    """
    Generate question answer pair from input text 
    @param text: text to generate eval set from
    @param chunk: chunk size to draw question from text
    @param logger: logger
    @return: dict with keys "question" and "answer"
    """

    logger.info("`Generating eval QA pair ...`")
    # Generate random starting index in the doc to draw question from
    num_of_chars = len(text)
    starting_index = random.randint(0, num_of_chars-chunk)
    sub_sequence = text[starting_index:starting_index+chunk]
    # Set up QAGenerationChain chain using GPT 3.5 as default
    #chain = QAGenerationChain.from_llm(ChatOpenAI(temperature=0))
    chain = QAGenerationChain.from_llm(AzureChatOpenAI(deployment_name="verifi-gpt-35-turbo", 
                              model_name="gpt-35-turbo", 
                              openai_api_base=os.getenv("OPENAI_API_BASE"),
                              openai_api_version=os.getenv("OPENAI_API_VERSION"),
                              openai_api_key=os.getenv("OPENAI_API_KEY"),
                              temperature=0))
    eval_set = []
    # Catch any QA generation errors and re-try until QA pair is generated
    awaiting_answer = True
    while awaiting_answer:
        try:
            qa_pair = chain.run(sub_sequence)
            eval_set.append(qa_pair)
            awaiting_answer = False
        except JSONDecodeError:
            logger.error("Error on question")
            starting_index = random.randint(0, num_of_chars-chunk)
            sub_sequence = text[starting_index:starting_index+chunk]
    eval_pair = list(itertools.chain.from_iterable(eval_set))
    return eval_pair


def split_texts(text, chunk_size, overlap, split_method, logger):
    """
    Split text into chunks
    @param text: text to split
    @param chunk_size: charecters per split
    @param overlap: charecter overlap between splits
    @param split_method: method used to split text
    @param logger: logger
    @return: list of str splits
    """

    logger.info("`Splitting doc ...`")
    if split_method == "RecursiveTextSplitter":
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                       chunk_overlap=overlap)
    elif split_method == "CharacterTextSplitter":
        text_splitter = CharacterTextSplitter(separator=" ",
                                              chunk_size=chunk_size,
                                              chunk_overlap=overlap)
    splits = text_splitter.split_text(text)
    return splits


def make_llm(model, logger):
    """
    Make LLM
    @param model: LLM to use
    @return: LLM
    """

    if model in ("gpt-3.5-turbo", "gpt-4"):
        llm = ChatOpenAI(model_name=model, temperature=0)
    elif model == "anthropic":
        llm = Anthropic(temperature=0)
    elif model == "Anthropic-100k":
        llm = Anthropic(model="claude-v1-100k",temperature=0)
    elif model == "vicuna-13b":
        llm = Replicate(model="replicate/vicuna-13b:e6d469c2b11008bb0e446c3e9629232f9674581224536851272c54871f84076e",
                input={"temperature": 0.75, "max_length": 3000, "top_p":0.25})
    elif model == "mosaic":
        llm = MosaicML(inject_instruction_format=True,model_kwargs={'do_sample': False, 'max_length': 3000})
    elif model == "falcon-7b":
        llm = Replicate(model="own-ai/oasst-falcon-7b-sft-top1-696:53032738862a3285f1517b2d2d92e82fe76dede87ba654956dd1e2eae4aa549c",
                       input={"temperature":0.75, "max_length": 3000, "top_p":0.25})
    elif model == "flan-t5-xl":
        llm = Replicate(model="replicate/flan-t5-xl:7a216605843d87f5426a10d2cc6940485a232336ed04d655ef86b91e020e9210",
                        input={"temperature": 0.75, "max_length": 3000, "top_p":0.25})
    elif model == "stablelm-tuned-alpha-7b":
        llm = Replicate(model="stability-ai/stablelm-tuned-alpha-7b:c49dae362cbaecd2ceabb5bd34fdb68413c4ff775111fea065d259d577757beb",
                        input={"temperature": 0.75, "max_length": 3000, "top_p":0.25})
    elif model == 'azure-gpt-3.5-turbo':
        #set_env_azure_openai()
        llm = AzureChatOpenAI(deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"), 
                              model_name=os.getenv("AZURE_OPENAI_MODEL_NAME"), 
                              openai_api_base=os.getenv("OPENAI_API_BASE"),
                              openai_api_version=os.getenv("OPENAI_API_VERSION"),
                              openai_api_key=os.getenv("OPENAI_API_KEY"),
                              temperature=0)
        
    logger.info(f'llm = {llm}')
    return llm

def make_retriever(splits, retriever_type, embeddings, num_neighbors, llm, logger):
    """
    Make document retriever
    @param splits: list of str splits
    @param retriever_type: retriever type
    @param embedding_type: embedding type
    @param num_neighbors: number of neighbors for retrieval
    @param _llm: model
    @param logger: logger
    @return: retriever
    """

    logger.info("`Making retriever ...`")
    # Set embeddings
    if embeddings == "OpenAI":
        embd = OpenAIEmbeddings()
    elif embeddings == "Azure OpenAI":
        # deployment name must be set to `text-embedding-ada-002` due to lack of deployment name field in langchain
        embd = AzureOpenAIEmbeddings(chunk_size=1)
    # Note: Still WIP (can't be selected by user yet)
    elif embeddings == "LlamaCppEmbeddings":
        embd = LlamaCppEmbeddings(model="replicate/vicuna-13b:e6d469c2b11008bb0e446c3e9629232f9674581224536851272c54871f84076e")
    # Note: Test
    elif embeddings == "Mosaic":
        embd = MosaicMLInstructorEmbeddings(query_instruction="Represent the query for retrieval: ")
    elif embeddings == "HuggingFace":
        embd = HuggingFaceEmbeddings()

    # Select retriever
    if retriever_type == "similarity-search":
        vectorstore = FAISS.from_texts(splits, embd)
        retriever = vectorstore.as_retriever(k=num_neighbors)
    elif retriever_type == "SVM":
        retriever = SVMRetriever.from_texts(splits, embd)
    elif retriever_type == "TF-IDF":
        retriever = TFIDFRetriever.from_texts(splits)
    elif retriever_type == "Anthropic-100k":
         retriever = llm
    return retriever

def make_chain(llm, retriever, retriever_type, model,logger):

    """
    Make retrieval chain
    @param llm: model
    @param retriever: retriever
    @param retriever_type: retriever type
    @return: QA chain
    """
    logger.info("Making chain")
    # Select prompt 
    if model == "vicuna-13b":
        # Note: Better answer quality using default prompt 
        # chain_type_kwargs = {"prompt": QA_CHAIN_PROMPT_LLAMA}
        chain_type_kwargs = {"prompt": QA_CHAIN_PROMPT}
    else: 
        chain_type_kwargs = {"prompt": QA_CHAIN_PROMPT}

    # Select model 
    if retriever_type == "Anthropic-100k":
        qa_chain = load_qa_chain(llm,chain_type="stuff",prompt=QA_CHAIN_PROMPT)
    else:
        qa_chain = RetrievalQA.from_chain_type(llm,
                                               chain_type="stuff",
                                               retriever=retriever,
                                               chain_type_kwargs=chain_type_kwargs,
                                               input_key="question")
    return qa_chain


def grade_model_answer(predicted_dataset, predictions, grade_answer_prompt, logger):
    """
    Grades the answer based on ground truth and model predictions.
    @param predicted_dataset: A list of dictionaries containing ground truth questions and answers.
    @param predictions: A list of dictionaries containing model predictions for the questions.
    @param grade_answer_prompt: The prompt level for the grading. Either "Fast" or "Full".
    @param logger: logger
    @return: A list of scores for the distilled answers.
    """

    logger.info("`Grading model answer ...`")
    if grade_answer_prompt == "Fast":
        prompt = GRADE_ANSWER_PROMPT_FAST
    elif grade_answer_prompt == "Descriptive w/ bias check":
        prompt = GRADE_ANSWER_PROMPT_BIAS_CHECK
    elif grade_answer_prompt == "OpenAI grading prompt":
        prompt = GRADE_ANSWER_PROMPT_OPENAI
    else:
        prompt = GRADE_ANSWER_PROMPT

    # Note: GPT-4 grader is advised by OAI 
   #eval_chain = QAEvalChain.from_llm(llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0),
   #                                   prompt=prompt)
    eval_chain = QAEvalChain.from_llm(llm=AzureChatOpenAI(deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"), 
                              model_name=os.getenv("AZURE_OPENAI_MODEL_NAME"), 
                              openai_api_base=os.getenv("OPENAI_API_BASE"),
                              openai_api_version=os.getenv("OPENAI_API_VERSION"),
                              openai_api_key=os.getenv("OPENAI_API_KEY"),
                              temperature=0),
                                      prompt=prompt)
    graded_outputs = eval_chain.evaluate(predicted_dataset,
                                         predictions,
                                         question_key="question",
                                         prediction_key="result")
    return graded_outputs


def grade_model_retrieval(gt_dataset, predictions, grade_docs_prompt, logger):
    """
    Grades the relevance of retrieved documents based on ground truth and model predictions.
    @param gt_dataset: list of dictionaries containing ground truth questions and answers.
    @param predictions: list of dictionaries containing model predictions for the questions
    @param grade_docs_prompt: prompt level for the grading.
    @return: list of scores for the retrieved documents.
    """

    logger.info("`Grading relevance of retrieved docs ...`")
    if grade_docs_prompt == "Fast":
        prompt = GRADE_DOCS_PROMPT_FAST
    else:
        prompt = GRADE_DOCS_PROMPT

    # Note: GPT-4 grader is advised by OAI
    #eval_chain = QAEvalChain.from_llm(llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0),
    #                                  prompt=prompt)
    eval_chain = QAEvalChain.from_llm(llm=AzureChatOpenAI(deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"), 
                              model_name=os.getenv("AZURE_OPENAI_MODEL_NAME"), 
                              openai_api_base=os.getenv("OPENAI_API_BASE"),
                              openai_api_version=os.getenv("OPENAI_API_VERSION"),
                              openai_api_key=os.getenv("OPENAI_API_KEY"),
                              temperature=0),
                                      prompt=prompt)
    graded_outputs = eval_chain.evaluate(gt_dataset,
                                         predictions,
                                         question_key="question",
                                         prediction_key="result")
    return graded_outputs


def run_eval(chain, retriever, eval_qa_pair, grade_prompt, retriever_type, num_neighbors, text, logger):
    """
    Runs evaluation on a model's performance on a given evaluation dataset.
    @param chain: Model chain used for answering questions
    @param retriever:  Document retriever used for retrieving relevant documents
    @param eval_set: List of dictionaries containing questions and corresponding ground truth answers
    @param grade_prompt: String prompt used for grading model's performance
    @param retriever_type: String specifying the type of retriever used
    @param num_neighbors: Number of neighbors to retrieve using the retriever
    @param text: full document text
    @return: A tuple of four items:
    - answers_grade: A dictionary containing scores for the model's answers.
    - retrieval_grade: A dictionary containing scores for the model's document retrieval.
    - latencies_list: A list of latencies in seconds for each question answered.
    - predictions_list: A list of dictionaries containing the model's predicted answers and relevant documents for each question.
    """

    logger.info("`Running eval ...`")
    predictions = []
    retrieved_docs = []
    gt_dataset = []
    latency = []

    # Get answer and log latency
    start_time = time.time()
    if retriever_type == "Anthropic-100k":
        docs=[Document(page_content=text)]
        answer = chain.run(input_documents=docs,question=eval_qa_pair["question"])
        predictions.append(
            {"question": eval_qa_pair["question"], "answer": eval_qa_pair["answer"], "result": answer})
    else :
        predictions.append(chain(eval_qa_pair))
    gt_dataset.append(eval_qa_pair)
    end_time = time.time()
    elapsed_time = end_time - start_time
    latency.append(elapsed_time)

    # Extract text from retrieved docs
    retrieved_doc_text = ""
    if retriever_type == "Anthropic-100k":
        retrieved_doc_text = "Doc %s: " % str(eval_qa_pair["answer"])
    else:
        docs = retriever.get_relevant_documents(eval_qa_pair["question"])
        for i, doc in enumerate(docs):
            retrieved_doc_text += "Doc %s: " % str(i+1) + \
                doc.page_content + " "

    # Log
    retrieved = {"question": eval_qa_pair["question"],
                 "answer": eval_qa_pair["answer"], "result": retrieved_doc_text}
    retrieved_docs.append(retrieved)

    # Grade
    graded_answers = grade_model_answer(
        gt_dataset, predictions, grade_prompt, logger)
    graded_retrieval = grade_model_retrieval(
        gt_dataset, retrieved_docs, grade_prompt, logger)
    return graded_answers, graded_retrieval, latency, predictions

#load_dotenv()

if os.environ.get("ENVIRONMENT") != "development":
    sentry_sdk.init(
    dsn="https://065aa152c4de4e14af9f9e7335c8eae4@o4505106202820608.ingest.sentry.io/4505106207735808",
    traces_sample_rate=1.0,
    )



app = FastAPI()

origins = [
    "http://localhost:3000",
    "localhost:3000",
    "https://evaluator-ui.vercel.app/"
    "https://evaluator-ui.vercel.app"
    "evaluator-ui.vercel.app/"
    "evaluator-ui.vercel.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Welcome to the Auto Evaluator!"}


def run_evaluator(
    files,
    num_eval_questions,
    chunk_chars,
    overlap,
    split_method,
    retriever_type,
    embeddings,
    model_version,
    grade_prompt,
    num_neighbors,
    test_dataset
):

    # Set up logging
    logging.config.fileConfig('logging.conf', disable_existing_loggers=False)
    logger = logging.getLogger(__name__)

    # Read content of files
    texts = []
    fnames = []
    for file in files:
        logger.info("Reading file: {}".format(file.filename))
        contents = file.file.read()
        # PDF file
        if file.content_type == 'application/pdf':
            logger.info("File {} is a PDF".format(file.filename))
            pdf_reader = pypdf.PdfReader(io.BytesIO(contents))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            texts.append(text)
            fnames.append(file.filename)
        # Text file
        elif file.content_type == 'text/plain':
            logger.info("File {} is a TXT".format(file.filename))
            texts.append(contents.decode())
            fnames.append(file.filename)
        else:
            logger.warning(
                "Unsupported file type for file: {}".format(file.filename))
    text = " ".join(texts)

    if retriever_type == "Anthropic-100k":
        splits = ""
        model_version = "Anthropic-100k"
    else:
        logger.info("Splitting texts")
        splits = split_texts(text, chunk_chars, overlap, split_method, logger)

    logger.info("Make LLM")
    llm = make_llm(model_version, logger)

    logger.info("Make retriever")
    retriever = make_retriever(
        splits, retriever_type, embeddings, num_neighbors, llm, logger)

    logger.info("Make chain")
    qa_chain = make_chain(llm, retriever, retriever_type, model_version, logger)

    for i in range(num_eval_questions):

        # Generate one question
        if i < len(test_dataset):
            eval_pair = test_dataset[i]
        else:
            eval_pair = generate_eval(text, 3000, logger)
            if len(eval_pair) == 0:
                # Error in eval generation
                continue
            else:
                # This returns a list, so we unpack to dict
                eval_pair = eval_pair[0]

        # Run eval
        graded_answers, graded_retrieval, latency, predictions = run_eval(
            qa_chain, retriever, eval_pair, grade_prompt, retriever_type, num_neighbors, text, logger)

        # Assemble output
        d = pd.DataFrame(predictions)
        d['answerScore'] = [g['text'] for g in graded_answers]
        d['retrievalScore'] = [g['text'] for g in graded_retrieval]
        d['latency'] = latency

        # Summary statistics
        d['answerScore'] = [{'score': 1 if "Incorrect" not in text else 0,
                             'justification': text} for text in d['answerScore']]
        d['retrievalScore'] = [{'score': 1 if "Incorrect" not in text else 0,
                                'justification': text} for text in d['retrievalScore']]

        # Convert dataframe to dict
        d_dict = d.to_dict('records')
        if len(d_dict) == 1:
            yield json.dumps({"data":  d_dict[0]})
        else:
            logger.warn(
                "A QA pair was not evaluated correctly. Skipping this pair.")


@app.post("/evaluator-stream")
async def create_response(
    files: List[UploadFile] = File(...),
    num_eval_questions: int = Form(5),
    chunk_chars: int = Form(1000),
    overlap: int = Form(100),
    split_method: str = Form("RecursiveTextSplitter"),
    retriever_type: str = Form("similarity-search"),
    embeddings: str = Form("OpenAI"),
    model_version: str = Form("gpt-3.5-turbo"),
    grade_prompt: str = Form("Fast"),
    num_neighbors: int = Form(3),
    test_dataset: str = Form("[]"),
):
    test_dataset = json.loads(test_dataset)
    return EventSourceResponse(run_evaluator(files, num_eval_questions, chunk_chars,
                                             overlap, split_method, retriever_type, embeddings, model_version, grade_prompt, num_neighbors, test_dataset), headers={"Content-Type": "text/event-stream", "Connection": "keep-alive", "Cache-Control": "no-cache"})
