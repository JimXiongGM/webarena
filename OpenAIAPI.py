import json
import os
import sqlite3
import threading
import time
from typing import List

import openai
import requests
from tenacity import retry, stop_after_attempt, wait_fixed, wait_random_exponential
from tqdm import tqdm

"""
export https_proxy=http://127.0.0.1:7893 http_proxy=http://127.0.0.1:7893 all_proxy=socks5://127.0.0.1:7893
"""
os.environ["https_proxy"] = "http://127.0.0.1:7893"
os.environ["http_proxy"] = "http://127.0.0.1:7893"

DEFAULT_KEY = os.environ.get("OPENAI_API_KEY", None)
assert DEFAULT_KEY is not None, "OPENAI_API_KEY is None"

DEFAULT_CHAT_MODEL = "gpt-3.5-turbo-16k"

client = openai.OpenAI(api_key=DEFAULT_KEY)


def get_header(api_key):
    header = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    return header


def gpt_completion(
    model="text-davinci-003",
    prompt="hello",
    temperature=0,
    max_tokens=100,
    top_p=1,
    frequency_penalty=0.0,
    presence_penalty=0.0,
    stop=None,  # ["\n"],
    retry=1,
    **kwargs,
):
    while retry > 0:
        try:
            response = client.Completion.create(
                model=model,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                stop=stop,
                **kwargs,
            )
            # res = response["choices"][0]["text"]
            return response
        except Exception as e:
            time.sleep(2)
            retry -= 1
            if retry == 0:
                raise e


@retry(wait=wait_random_exponential(min=5, max=30), stop=stop_after_attempt(3))
def chatgpt(
    prompt="Hello!",
    system_content="You are an AI assistant.",
    messages=None,
    model=None,
    temperature=0,
    top_p=1,
    n=1,
    stop=None,  # ["\n"],
    max_tokens=256,
    presence_penalty=0,
    frequency_penalty=0,
    logit_bias={},
    **kwargs,
):
    """
    role:
        The role of the author of this message. One of `system`, `user`, or `assistant`.
    temperature:
        What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.
        We generally recommend altering this or `top_p` but not both.
    top_p:
        An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens comprising the top 10% probability mass are considered.
        We generally recommend altering this or `temperature` but not both.

    messages as history usage:
        history = [{"role": "system", "content": "You are an AI assistant."}]

        inp = "Hello!"
        history.append({"role": "user", "content": inp})
        response = chatgpt(messages=history)
        out = response["choices"][0]["message"]["content"]
        history.append({"role": "assistant", "content": out})
        print(json.dumps(history,ensure_ascii=False,indent=4))
    """

    # openkey.cloud
    # openai.api_base = "https://openkey.cloud/v1"
    assert model is not None, "model name is None"

    messages = (
        messages
        if messages
        else [
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt},
        ]
    )

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        n=n,
        stop=stop,
        max_tokens=max_tokens,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        logit_bias=logit_bias,
        # **kwargs,
    )
    # content = response["choices"][0]["message"]["content"]
    response = json.loads(response.json())
    return response

@retry(wait=wait_random_exponential(min=5, max=30), stop=stop_after_attempt(3))
def chatgpt_chn(
    prompt="Hello!",
    system_content="You are an AI assistant.",
    messages=None,
    model=None,
    temperature=0,
    top_p=1,
    n=1,
    stop=None,  # ["\n"],
    max_tokens=256,
    **kwargs
):
    os.environ.pop("https_proxy", None)
    os.environ.pop("http_proxy", None)
    # env: justsong
    assert model is not None, "model name is None"
    _key = os.environ.get("ONE_API_KEY", None)
    assert _key is not None, "ONE_API_KEY is None"
    headers = {
        "Authorization": "Bearer " + _key.strip(),
        "Content-Type": "application/json",
    }
    messages = (
        messages
        if messages
        else [
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt},
        ]
    )

    json_data = {
        "model": model,
        "messages": messages,
        "stream": False,
        "temperature": temperature,
        "top_p": top_p,
        "n": n,
        "stop": stop,
        "max_tokens": max_tokens,
    }

    response = requests.post(
        "https://wdapi5.61798.cn/v1/chat/completions",
        headers=headers,
        json=json_data,
    )
    response = response.json()
    return response


def check_limit(api_key):
    try:
        gpt_completion(prompt="Hello!", model="text-ada-001", api_key=api_key, retry=1)
    except openai.RateLimitError:
        return False
    return True


def check_api_keys(api_keys):
    for idx, key in enumerate(
        tqdm(api_keys, dynamic_ncols=True, colour="green", desc="Checking API keys")
    ):
        if check_limit(key):
            print("Good:\t", idx, key)


# 创建一个线程本地存储对象
thread_local = threading.local()


def get_sqlite_client():
    if not hasattr(thread_local, "cache_sql_client"):
        # 如果当前线程没有连接，则为它创建一个
        os.makedirs("database/cache_vector_query", exist_ok=True)
        cache_db_path = "database/cache_vector_query/local_cache.db"
        thread_local.cache_sql_client = sqlite3.connect(cache_db_path)
        cursor = thread_local.cache_sql_client.cursor()
        cursor.execute(
            """CREATE TABLE IF NOT EXISTS vec_cache (
                name TEXT PRIMARY KEY,
                vec TEXT NOT NULL
            );"""
        )
        thread_local.cache_sql_client.commit()
    return thread_local.cache_sql_client


def get_vec_cache(name):
    # 获取当前线程的 SQLite 客户端
    sql_client = get_sqlite_client()
    cursor = sql_client.cursor()
    cursor.execute(
        """SELECT vec FROM vec_cache WHERE name=?;""",
        (name,),
    )
    res = cursor.fetchone()
    if res:
        return json.loads(res[0])
    return None


def insert_vec_cache(name, vec):
    sql_client = get_sqlite_client()
    cursor = sql_client.cursor()
    if get_vec_cache(name):
        return
    if not isinstance(vec, str):
        vec = json.dumps(vec)
    cursor.execute(
        """INSERT INTO vec_cache (name, vec) VALUES (?, ?);""",
        (name, vec),
    )
    sql_client.commit()


@retry(wait=wait_fixed(10), stop=stop_after_attempt(2))
def get_embedding(
    text: str,
    model="text-embedding-ada-002",
) -> list[float]:
    text_unikey = text + model
    res = get_vec_cache(text_unikey)
    if res:
        assert type(res) == list
        return res
    res = openai.embeddings.create(input=[text], model=model).data[0].embedding
    insert_vec_cache(text_unikey, res)
    return res


# @retry(wait=wait_fixed(10), stop=stop_after_attempt(2))
def get_embedding_batch(
    texts: List[str],
    model="text-embedding-ada-002",
) -> list[float]:
    unseen_texts: List[str] = []
    for text in texts:
        cache = get_vec_cache(text + model)
        if cache is None:
            unseen_texts.append(text)

    if unseen_texts:
        req = openai.embeddings.create(input=unseen_texts, model=model)
        vec_batch = [i.embedding for i in req.data]
        assert len(vec_batch) == len(unseen_texts)
        for unseen_text, vec in zip(unseen_texts, vec_batch):
            insert_vec_cache(unseen_text + model, vec)

    res = [get_vec_cache(text + model) for text in texts]
    assert None not in res
    return res


def test():
    text = "what purpose did seasonal monsoon winds have on trade?"
    res = gpt_completion(prompt=text, model="text-curie-001")
    print(res)


def test_embedding():
    text = "hello!! how are you? I am fine."
    res = get_embedding(text)
    print(res)


def test_chatgpt():
    text = "who are you?"
    res = chatgpt(prompt=text, model="gpt-3.5-turbo", timeout=5, max_tokens=32)
    print(res)


chatgpt_tok = None


def chatgpt_tokenize(text):
    global chatgpt_tok
    if chatgpt_tok is None:
        import tiktoken

        chatgpt_tok = tiktoken.encoding_for_model("gpt-3.5-turbo")
    res = chatgpt_tok.encode(text)
    return res


embedding_tok = None


def embedding_tokenize(text):
    global embedding_tok
    if embedding_tok is None:
        import tiktoken

        embedding_tok = tiktoken.encoding_for_model("text-embedding-ada-002")
    res = embedding_tok.encode(text)
    return res


if __name__ == "__main__":
    # check_api_keys(api_keys=APIKEYS)
    print(chatgpt(prompt="what is your name?", model="gpt-3.5-turbo-16k-0613"))
    embedding = get_embedding("Your text goes here 123", model="text-embedding-ada-002")
    # embedding = get_embedding_batch(
    #     ["Your text goes here 12322", "Your text goes here 222"],
    #     model="text-embedding-ada-002",
    # )
    # print(len(embedding))
    # test_embedding()

    chatgpt_chn(
        prompt="你好！你叫什么？",
        model="gpt-4-1106-preview",
        max_tokens=32,
    )
