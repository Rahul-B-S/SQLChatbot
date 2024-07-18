from flask import Flask, render_template,request
from main import  chain, history

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('chat.html', doc_name = "World Database")


@app.route("/get", methods=["GET", "POST"])

def chat():
    question = request.form["question"]
    response = chain.invoke({"question": question,"messages":history.messages})
    return response


def get_Chat_response(text):
    pass

if __name__ == '__main__':
    app.run()