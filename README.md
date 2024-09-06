# Exploration of Approaches to Counter Hate Speech: The Case of Sexist Speech

Exploring methods for generating counter-speech against online sexism and hate-speech. 

## Introduction

This repo is the code implementation for "[Exploration of Approaches to Counter Hate Speech: The Case of Sexist Speech](https://something)".

This project focuses on exploring methods for **automated counter-speech generation**, which is, for a given hate-speech, to generate a countering response intending to combat the harmful messages, is non-toxic and grammatically acceptable.

Tested models:

[Generate, Prune, Select: A Pipeline for Counterspeech Generation against Online Hate Speech](https://github.com/WanzhengZhu/GPS) (ACL-IJCNLP Findings 2021)

[CounterGeDi: A controllable approach to generate polite, detoxified and emotional counterspeech](https://github.com/hate-alert/CounterGEDI) (IJCAI 2022)

## Project Structure

```
│./data
└──/...          ---------------------------------------- Sources for these datasets -> see Data section
└──/Custom       ---------------------------------------- combined dataset used for training and testing (pre-processed)
│
│./evaluation
└──/...          ---------------------------------------- Evaluation results
│
│./models
└──/bart_CONAN   ---------------------------------------- fine-tuned BART models
└──/gpt2_CONAN   ---------------------------------------- fine-tuned GPT2 models
│
│./predictions
└──/...          ---------------------------------------- Model predictions
│
│./NoteBooks
│./.py files
```
## Data

Reddit, Gab: Please download the data from [A Benchmark Dataset for Learning to Intervene in Online Hate Speech](https://github.com/jing-qian/A-Benchmark-Dataset-for-Learning-to-Intervene-in-Online-Hate-Speech/tree/master/data) (EMNLP2019) 

CONAN: Please download the data from [CONAN - COunter NArratives through Nichesourcing: a Multilingual Dataset of Responses to Fight Online Hate Speech](https://github.com/marcoguerini/CONAN) (ACL2019) 

## Add your files

- [ ] [Create](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#create-a-file) or [upload](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#upload-a-file) files
- [ ] [Add files using the command line](https://docs.gitlab.com/ee/gitlab-basics/add-file.html#add-a-file-using-the-command-line) or push an existing Git repository with the following command:

```
cd existing_repo
git remote add origin https://gitlab.lrz.de/masters-thesis-daryna/counter-sexist-hate-speech.git
git branch -M main
git push -uf origin main
```


***

