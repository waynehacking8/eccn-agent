"""
Script to generate embeddings for converted_ccl_data_checked.json using Azure OpenAI
參考 openai_client.py 的設計，並使用用戶指定的 endpoint/model
將10.75.120.140 gbb-ai-advantech-rd.openai.azure.com加入到C:/Windows/System32/drivers/etc/hosts此份文件中並儲存，在Windows環境才能順利使用
"""
import json
import pickle
import os
from typing import List, Dict, Any
from openai import AzureOpenAI
import tiktoken
from dotenv import load_dotenv

class ECCNEmbeddingClient:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv('AZURE_OPENAI_API_KEY')
        self.endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
        self.api_version = os.getenv('AZURE_OPENAI_API_VERSION', '2024-02-01')
        self.model_name = os.getenv('AZURE_OPENAI_EMBEDDING_MODEL', 'text-embedding-ada-002-for-GBB')
        if not self.api_key or not self.endpoint:
            raise ValueError('請在 .env 檔案中設置 AZURE_OPENAI_API_KEY 與 AZURE_OPENAI_ENDPOINT')
        self.client = AzureOpenAI(
            api_key=self.api_key,
            api_version=self.api_version,
            azure_endpoint=self.endpoint,
        )

    def split_text_into_chunks(self, text: str, max_tokens: int = 7000) -> list:
        max_chars = max_tokens * 4
        if len(text) <= max_chars:
            return [text]
        chunks = []
        current_pos = 0
        while current_pos < len(text):
            end_pos = current_pos + max_chars
            if end_pos >= len(text):
                chunks.append(text[current_pos:])
                break
            chunk_end = end_pos
            for i in range(end_pos, current_pos + max_chars//2, -1):
                if text[i] in '.。\n\r':
                    chunk_end = i + 1
                    break
                elif text[i] in ' \t':
                    chunk_end = i
                    break
            chunks.append(text[current_pos:chunk_end])
            current_pos = chunk_end
        return chunks

    def generate_embedding_for_text(self, text: str) -> list:
        embedding = self.client.embeddings.create(
            input=[text],
            model=self.model_name
        ).data[0].embedding
        return embedding

    def generate_embeddings_for_data(self, data: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        print(f"Generating embeddings for {len(data)} entries...")
        all_entries = {}
        successful_embeddings = 0
        failed_embeddings = 0
        for i, entry in enumerate(data):
            if i % 10 == 0:
                print(f"Processing entry {i+1}/{len(data)}: {entry['eccn_code']}")
            try:
                text_chunks = self.split_text_into_chunks(entry['text'], max_tokens=7000)
                if len(text_chunks) > 1:
                    print(f" {entry['eccn_code']} 分割為 {len(text_chunks)} 段，將拆成多筆embedding")
                for idx, chunk in enumerate(text_chunks):
                    code = entry['eccn_code'] + (f'_chunk{idx+1}' if len(text_chunks) > 1 else '')
                    try:
                        embedding = self.generate_embedding_for_text(chunk)
                        all_entries[code] = {
                            'text': chunk,
                            'embedding_array': embedding
                        }
                        successful_embeddings += 1
                    except Exception as e:
                        print(f"Error generating embedding for {code}: {str(e)}")
                        all_entries[code] = {
                            'text': chunk,
                            'embedding_array': []
                        }
                        failed_embeddings += 1
            except Exception as e:
                print(f"Error splitting/embedding for {entry['eccn_code']}: {str(e)}")
                failed_embeddings += 1
        print(f"\nEmbedding generation complete:")
        print(f"Successful: {successful_embeddings}")
        print(f"Failed: {failed_embeddings}")
        return all_entries

def load_converted_data(file_path: str) -> List[Dict[str, Any]]:
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_as_pickle(data: Dict[str, Dict[str, Any]], output_path: str):
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Saved embeddings to {output_path}")

def main():
    input_file = os.path.join(os.path.dirname(__file__), 'converted_ccl_data_checked.json')
    output_pkl = os.path.join(os.path.dirname(__file__), 'eccn_embeddings_checked.pkl')
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} does not exist!")
        return
    print("Loading converted CCL data (checked)...")
    converted_data = load_converted_data(input_file)
    print(f"Loaded {len(converted_data)} ECCN entries")
    print("Generating embeddings...")
    client = ECCNEmbeddingClient()
    data_with_embeddings = client.generate_embeddings_for_data(converted_data)
    print("Saving as pickle file...")
    save_as_pickle(data_with_embeddings, output_pkl)
    print(f"Process completed! New PKL file created at: {output_pkl}")
    print(f"Total entries: {len(data_with_embeddings)}")
    if data_with_embeddings:
        first_key = next(iter(data_with_embeddings))
        sample_entry = data_with_embeddings[first_key]
        print(f"\nSample entry:")
        print(f"ECCN Code: {first_key}")
        print(f"Text length: {len(sample_entry['text'])} characters")
        print(f"Embedding dimensions: {len(sample_entry['embedding_array'])}")

if __name__ == "__main__":
    main() 