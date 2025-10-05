import requests

url = "http://127.0.0.1:5000/answer"

while True:

    question = input("Enter your question (or type 'exit' to quit): ")


    if question.lower() == 'exit':
        print("Exiting the application.")
        break


    data = {
        "question": question
    }

    try:

        response = requests.post(url, json=data)


        if response.status_code == 200:

            response_data = response.json()
            print(f"Answer: {response_data['answer']}")
            print(f"Top Snippets: {response_data['top_snippets']}")
            print(f"Search Method: {response_data['search_method']}")
            print(f"Processing Time: {response_data['processing_time']} seconds")
        else:
            print(f"Error: {response.status_code} - {response.text}")

    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
