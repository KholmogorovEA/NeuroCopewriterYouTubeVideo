import os
import logging
from dotenv import load_dotenv
import openai
import re
import requests
from openai import OpenAI
import subprocess
from pathlib import Path
import json
from tqdm.auto import tqdm
from urllib.request import urlopen
import openai
import shutil
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI   
from langchain_community.llms import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain.schema import Document
from pydub import AudioSegment
import os
import shutil
import time


load_dotenv()
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)
openai.api_key = os.getenv('OPENAI_API_KEY')
from openai import OpenAI
client = OpenAI()


import subprocess
import os

os.makedirs('tmp', exist_ok=True)

yt_urls = ['https://www.youtube.com/watch?v=Uqp-pzGMjlU']
url = yt_urls[0]

# Получаем имя файла
file_name = subprocess.run(
    ['yt-dlp', url, '-f', 'bestaudio[ext=m4a]', '--get-filename', '-o', 'tmp/%(title)s.m4a'],
    capture_output=True, text=True
).stdout.strip()

# Загружаем аудио
subprocess.run(
    ['yt-dlp', url, '-f', 'bestaudio[ext=m4a]', '-o', 'tmp/%(title)s.m4a']
)

print(f"Файл сохранен как: {file_name}")







def transcribe_audio_whisper_chunked(audio_path, file_title, save_folder_path, max_duration=3 * 60 * 1000):  # 3 минуты
    """
    Транскрибирует аудиофайл по частям, чтобы соответствовать ограничениям размера API.
    """
    os.makedirs(save_folder_path, exist_ok=True)
    audio = AudioSegment.from_file(audio_path)
    temp_dir = os.path.join(save_folder_path, "temp_audio_chunks")
    os.makedirs(temp_dir, exist_ok=True)

    current_start_time = 0
    chunk_index = 1
    transcriptions = []

    while current_start_time < len(audio):
        chunk = audio[current_start_time:current_start_time + max_duration]
        chunk_name = f"chunk_{chunk_index}.wav"
        chunk_path = os.path.join(temp_dir, chunk_name)
        chunk.export(chunk_path, format="wav")

        if os.path.getsize(chunk_path) > 26214400:  # 25 MB
            print(f"Chunk {chunk_index} exceeds the maximum size limit for the API. Trying a smaller duration...")
            max_duration = int(max_duration * 0.9)
            os.remove(chunk_path)
            continue

        with open(chunk_path, "rb") as src_file:
            print(f"Transcribing {chunk_name}...")
            try:
                # Вызов API для транскрибации
                transcription = client.audio.transcriptions.create(model="whisper-1", file=src_file)
                transcriptions.append(transcription.text)  # Используем атрибут .text для получения текста
            except Exception as e:
                print(f"An error occurred: {e}")
                break

        os.remove(chunk_path)
        current_start_time += max_duration
        chunk_index += 1

    shutil.rmtree(temp_dir)

    
    result_path = os.path.join(save_folder_path, f"{file_title}.txt")
    with open(result_path, "w") as txt_file:
        txt_file.write("\n".join(transcriptions))
    
    print(f"Transcription saved to {result_path}")


# Задаем исходные параметры и вызываем функцию
audio_path = 'Графы： алгоритмы и структуры данных на Python.m4a'
file_title = 'Графы_Алгоритмы_и_Структуры_Данных_на_Python'
save_folder_path = 'C:\\Users\\Evgenii\\Desktop\\bot_rezumer_video'

# Вызов функции для транскрибации аудиофайла
# transcribe_audio_whisper_chunked(audio_path, file_title, save_folder_path)





system = 'Вы гений текста, копирайтинга, писательства. Ваша задача распознать разделы в тексте и разбить его на эти разделы сохраняя весь текст на 100%'
user = 'Пожалуйста, давайте подумаем шаг за шагом: Подумайте, какие разделы в тексте вы можете распознать и какое название по смыслу можно дать каждому разделу. Далее напишите ответ по всему предыдущему ответу в порядке: ## Название раздела, после чего весь текст, относящийся к этому разделу. Текст:' 

temperature = 0
chunk_size = 6000 

# Функция дробления текста на чанки
def split_text(txt_file, chunk_size=chunk_size):
    source_chunks = []
    splitter = RecursiveCharacterTextSplitter(separators=['\n', '\n\n', '. '], chunk_size=chunk_size, chunk_overlap=0)

    for chunk in splitter.split_text(txt_file):
        source_chunks.append(Document(page_content=chunk, metadata={}))

    print(f'\n\nТекст разбит на {len(source_chunks)} чанков.')

    return source_chunks


def answer_index(system: str, user: str, chunk, temp=temperature, model='gpt-3.5-turbo-16k') -> str:

    messages = [
        {'role': 'system', 'content': system},
        {'role': 'user', 'content': user + f'{chunk}'}
    ]

    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temp
    )

    # Вывод количества токенов отключен
    # print(f'\n====================\n\n{num_tokens_from_messages(messages)} токенов будет использовано на чанк\n\n')
    answer = completion.choices[0].message.content

    return answer





def process_one_file(file_path, system, user):
    with open(file_path, 'r') as txt_file:
        text = txt_file.read()
    
    source_chunks = split_text(text)
    processed_text = ''
    unprocessed_text = ''

    for chunk in source_chunks:
        attempt = 0
        answer = ''  

        while attempt < 3:
            try:
                print(f'Обрабатываем часть: {chunk}')  
                answer = answer_index(system, user, chunk.page_content)
                print(f"Ответ от answer_index: {answer}")  
                
                if answer: 
                    break  
                else:
                    raise ValueError("Ответ пустой!")  

            except Exception as e:
                attempt += 1  
                print(f'\nПопытка {attempt} не удалась из-за ошибки: {str(e)}')
                
                if attempt < 3:  
                    time.sleep(5)  
                else:
                    answer = ''  
                    print(f'\nОбработка элемента {chunk} не удалась после 3 попыток')
                    unprocessed_text += f'{chunk}\n\n'

        processed_text += f'{answer}\n\n'  
        print(f'ЭТО РЕЗУЛЬТАТ РАБОТЫ PROCESS ONE FILE---------------------------------------------------{answer}')  

    return processed_text, unprocessed_text

file_path = "Графы_Алгоритмы_и_Структуры_Данных_на_Python.txt"

processed_text, unprocessed_text = process_one_file(file_path, system, user)




headers_to_split_on = [
    ("##", "Header 2")
    ]

markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)


md_header_splits = markdown_splitter.split_text(processed_text)
print(len(md_header_splits))

system1 = "Вы гений копирайтинга, эксперт в программировании на пайтон и в теме Графы, алгоритмы и структуры данных. Вы получаете раздел необработанного текста по определенной теме. Нужно из этого текста выделить самую суть, только самое важное, сохранив все нужные подробности и детали, но убрав всю (воду) и слова (предложения), не несущие смысловой нагрузки." 
user1 = "Из данного текста выдели только ценную с точки зрения темы (графы, алгоритмы и структуры данных) информацию. Удали всю (воду). В итоге у тебя должен получится раздел для методички по указанной теме. Опирайся только на данный тебе текст, не придумывай ничего от себя. Ответ нужен в формате ## Название раздела, и далее выделенная тобой ценная информация из текста. Если в тексте не содержится ценной информации, то оставь только  название раздела, например:"

temperature1 = 0 





def process_documents(documents, system1, user1, temperature):
    """
    Функция принимает чанки, system, user, temperature для модели.
    Она обрабатывает каждый документ, используя модель GPT, конкатенирует результаты в один текст и сохраняет в файл .txt.
    В итоге мы получаем методичку по лекции.
    """
    processed_text_for_handbook = ""  

    for document in documents:
      
        metadata_str = "\n".join([f"{key}: {value}" for key, value in document.metadata.items()])
   
        chunk_with_metadata = f"{metadata_str}\n\n{document.page_content}"

        answer = answer_index(system1, user1, chunk_with_metadata, temperature, model='gpt-4-0613')
   
        processed_text_for_handbook += f"{answer}\n\n"

 
    with open('processed_documents.txt', 'w', encoding='utf-8') as f:
        f.write(processed_text_for_handbook)
    print('00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000')

    return 'processed_documents.txt'


file_path = process_documents(md_header_splits, system1, user1, temperature)
print(f"Обработанный текст сохранен в файле: {file_path}")


# Чтение и вывод содержимого методички:
with open(file_path, 'r', encoding='utf-8') as f:
    processed_text = f.read()

print(f' ЭТО МЕТОДИЧКА------------------------------------------------------------------ {processed_text}')






embeddings = OpenAIEmbeddings()


db = FAISS.from_documents(md_header_splits, embeddings)
print(db)

system_for_NA = """Ты - преподаватель, эксперт по теме 'Графы, алгоритмы и структуры данных.'
                  Твоя задача - ответить студенту на вопрос только на основе представленных тебе документов, не добавляя ничего от себя."""



def answer_neuro_assist(system_for_NA, topic, search_index):

    docs = search_index.similarity_search(topic, k=3)
    # if verbose: print('\n ===========================================: ')
    message_content = re.sub(r'\n{2}', ' ', '\n '.join([f'\nОтрывок документа №{i+1}\n=====================' + doc.page_content + '\n' for i, doc in enumerate(docs)]))
    # if verbose: print('message_content :\n ======================================== \n', message_content)

    messages = [
        {"role": "system", "content": system_for_NA},
        {"role": "user", "content": f"Ответь на вопрос студента. Не упоминай отрывки документов с информацией для ответа студенту в ответе. Документ с информацией для ответа студенту: {message_content}\n\nВопрос студента: \n{topic}"}
    ]

    # if verbose: print('\n ===========================================: ')

    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0
    )
    answer = completion.choices[0].message.content
    return answer 




topic="Что такое графы"
ans=answer_neuro_assist(system_for_NA, topic, db)
print(ans)