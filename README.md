# Manipulation Detection

This repository is devoted to the thesis on the topic of the Development of a tool to automate the detection of manipulative methods in political discourse. 

The repository consists of files required to host a website as well as labeled documents and the analysis of final output.

Files with .docx extension are the documents used while working on the thesis. The ones with _ground_truth_ in the name are manually-labeled. The ones with _chatgpt_final_ in the name are automatically labeled. File Evaluation and metrics.ipynb contains the calculated metrics for automatically labeled texts.

For hosting a website you need files app.py, base.html.j2, index.html.j2, model.py, requirements.txt and credentials.py. The repository contains all of them except for credentials.py. It was done for safety reasons, and to host a website you need to get a key to OpenAI API (ChatGPT model). If you have the key, here is how you host a website:

1. Pack all the necessary files in one folder
2. Create subfolder 'templates' and put files _.html.j2 in it
3. Create subfolder 'analyzed_files'
4. Open the Terminal
5. Specify the path to the folder by using cd /path/
6. Input pip install -r requirements.txt
7. Input python app.py
8. Terminal will give you a link to the website
