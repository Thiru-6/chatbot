import google.generativeai as genai
import dotenv
import os
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

dotenv.load_dotenv()

class RagFunctions:
    '''
    Initializes the model when we call the class  so that the model is not initialized whenever the generate response function is called
    '''
    def __init__(self):
        self.api_key = os.getenv('GEMINI_API_KEY')
        self.model_name = "nomic-ai/nomic-embed-text-v1.5"
        self.gemini_model_name = "gemini-1.5-pro-latest"
        self.generation_config = {
            'temperature' : 0.95,
            'top_p' : 1,
            'top_k' : 1,
            'max_output_tokens' : 8192
        }
        self.db_path = "vectorstores/chroma"

        genai.configure(api_key = self.api_key)
        self.model = genai.GenerativeModel(
            model_name = self.gemini_model_name,
            generation_config = self.generation_config
        )

    
    def get_embeddings(self):
        embeddings = HuggingFaceEmbeddings(
            model_name = self.model_name,
            model_kwargs = {'device' : 'cpu' , 'trust_remote_code' : True}
        )
        return embeddings
    
    def get_query_documents(self , query):
        embeddings = self.get_embeddings()
        db = Chroma(persist_directory=self.db_path , embedding_function= embeddings)
        search_result = db.search(query , search_type="similarity")
        if 'documents' in search_result and isinstance(search_result['documents'], list):
            context_list = []
            for document in search_result['documents']:
                # Extract page_content attribute from each document
                page_content = document.get('page_content', '')  # Assuming page_content is a key in the document
                context_list.append(page_content)

            return context_list
        else:
            return None
        
    def generate_response(self , query):
        context = self.get_query_documents(query)
        prompt = f"""
        Using the following pieces of information try to give the best answer for the question asked by the user.
        Give the correct response if the question is related to medical or biology, if not then just tell that the provided query is not related to medical or biology.

        Context : {context}
        Question : {query}

        Only return useful responses and nothing else.
        Helpful Response:
        """
        response = self.model.generate_content([prompt])

        return response.text
    
        