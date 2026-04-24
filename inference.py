
# Требования: Python 3.8+, библиотека requests, запущенный Ollama.


import requests
from typing import List, Dict

OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:0.5b"

def send_query(prompt: str, model: str = MODEL_NAME) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False  # потоковый вывод для получения полного ответа
    }
    try:
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        return data.get("response", "")
    except requests.exceptions.RequestException as e:
        return f"[Ошибка подключения: {e}]"


def run_batch_queries(queries: List[str], model: str = MODEL_NAME) -> List[Dict[str, str]]:
   
    results = []
    total = len(queries)
    for i, query in enumerate(queries, 1):
        print(f"[{i}/{total}] Отправляю запрос: {query[:50]}...")
        response = send_query(query, model)
        results.append({"query": query, "response": response})
        print(f"Готово ({len(response)} символов)\n")
    return results


def save_report(results: List[Dict[str, str]], filename: str = "inference_report.md") -> None:

    with open(filename, "w", encoding="utf-8") as f:
        f.write("# Отчёт инференса Qwen2.5:0.5B\n\n")
        f.write("| Запрос | Ответ LLM |\n")
        f.write("|--------|-----------|\n")

        for item in results:
            query_clean = item["query"].replace("|", "\\|").replace("\n", "<br>")
            response_clean = item["response"].replace("|", "\\|").replace("\n", "<br>")
            f.write(f"| {query_clean} | {response_clean} |\n")
    print(f"Отчёт сохранён в {filename}")


def main() -> None:

    print("Запуск инференса модели", MODEL_NAME)
    print("URL сервера:", OLLAMA_API_URL)

    # 10 запросов 
    test_queries = [
        "Какова длина экватора?",
        "Напиши формулу геометрической прогрессии",
        "Какая столица Франции?",
        "Сравни языки C++ и Python",
        "Напиши короткое стихотворение",
        "Что такое контейнеризация и зачем она нужна?",
        "Как пользоваться PowerShell?",
        "Напиши реализацию dequeu на С++",
        "Что такое overfitting в машинном обучении?",
        "Что такое триангуляция?"
    ]

    results = run_batch_queries(test_queries)
    save_report(results)
    print("Все запросы обработаны. Лабораторная работа готова к сдаче!")


if __name__ == "__main__":
    main()