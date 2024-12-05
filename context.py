from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings,SimpleDirectoryReader,VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from transformers import AutoModelForCausalLM,AutoTokenizer
from llama_index.readers.file import PDFReader
import tempfile
import os

class RAGSystem:
    def __init__(self):
        self._initialize_settings()
        self._initialize_model()
        self.index=None

    def _initialize_settings(self):
        Settings.embed_model=HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        Settings.llm=None
        Settings.chunk_size=256   #here we are assuning 5000 chunck we are doing 5000/256=~20 so we diving documents into 20 chunks
        Settings.chunk_overlap=15  


    def _initialize_model(self):
        self.model_name="Qwen/Qwen2.5-1.5B-Instruct"
        self.model=AutoModelForCausalLM.from_pretrained(
                           self.model_name,
                           trust_remote_code=False,
                           revision="main",
                           #device_map='cuda:0'  who are having gpu can use
                           )
         #load tokenizer
        tokenizer=AutoTokenizer.from_pretrained(self.model_name,use_fast=True)  


    def process_pdf(self,file_content):
        try:
            with tempfile.NamedTemporaryFile(delete=False,suffix=".pdf") as tmp_file:
                tmp_file.write(file_content)
                tmp_path=tmp_file.name

        #Read PDF
            reader=PDFReader()
            documents=reader.load(tmp_path)

            self.index=VectorStoreIndex.from_documents(documents)

            #os clean
            os.unlike(tmp_path)
            return True

        except Exception as e:
            print(f"Error processing pdf: {str(e)}")  
            return False

    def get_query_engine(self,top_k=2):
        if not self.index:
            raise ValueError("No index available.Please process a pdf file")

        
        retriever=VectorIndexRetriever(
           index=self.index,
           similarity_top_k=top_k,
           )

        return RetrieverQueryEngine(
           retriever=retriever,
           node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.5)]#50% similar to it will retrived by query engine
           ) 


    def _create_prompt(self,context,query):
        return f"""I am an AI assiantant tasked with answering question based on the provided PDF content.
        please analyze the following except from PDF and answer the question
        PDF content:
        {context}

        Question:{query}


        Instructions:

         
        -Answer only based on the information provided in the PDF content above.
        -If the answer cannot be found in the provided content,say I cannot find the answer to the question and provide a PDF documnets
        -Be concise and specifice
        -Include relevant quote or references from the PDF when applicable
        Answer:"""  

    def generate_response(self,query_engine,query):
        try:
            response=query_engine.query(query)

            context=""

            for node in  response.source_nodes[:2]:
                context=context+f"{node.text}\n\n"
                #print(context)
            if not context.strip():
                return "No relvant information from PDF document"

            #create a prompt and generating a response 
            prompt=self._create_prompt(context,query)

            inputs=self.tokenizer(prompt,return_tensors='pt')
            outputs=self.model.generate(
                                   input_ids=inputs["input_ids"],
                                   max_new_tokens=512,
                                   num_return_sequences=1,
                                   temperature=0.3,
                                   top_p=0.9,
                                   do_sample=True,
                                   repetition_penalty=1.2)
            #print(tokenizer.batch_decode(outputs)[0])  
            response_text=self.tokenizer.decode(outputs[0],skip_special_tokens=True)
            if "Answer" in response_text:
                response_text=response_text.split("Answer:")[-1].strip()
            return response_text if response_text else "unable to generate a response from PDF documents"    
        except Exception as e:
            print(f"Error generating a response:{str(e)}")
            return f"Error processing your question:{str(e)}"
