from __future__ import unicode_literals, print_function, division
import records
from library.table import Table
from library.query import Query
import pandas as pd
import nltk


class DataConversionUtil:
    """Class is responsible for converting all the sql structured output to plain text sql queries"""

    def __init__(self):
        self.table_map = {}  # key is table_id, value is all the table data

    def build_table_mapping(self, dataset):
        """Reads the tables file and creates a dictionary with table id as key and all other data as value"""
        tables = pd.read_json("data/" + dataset + ".tables.jsonl", lines=True)
        data = pd.DataFrame()
        for index, line in tables.iterrows():
            self.table_map[line["id"]] = line
            line["tokenized_header"] = []
            for column_header in line["header"]:
                line["tokenized_header"].append(self.tokenize_document(column_header))
            line_df = pd.DataFrame(line)
            line_df = line_df.transpose()
            data = data.append(line_df)
        self.save_dataframe(data, "data/tokenized_" + dataset + ".tables.jsonl")

    def get_query_from_json(self, json_line):
        """Returns a Query object for the json input and returns the table object as well"""
        q = Query.from_dict(json_line["sql"])
        t_id = json_line["table_id"]
        table = self.table_map[t_id]
        t = Table("", table["header"], table["types"], table["rows"])
        return t, q

    @staticmethod
    def execute_query(table, query):
        """Executes a query on the sqlite training db. Only for testing purposes"""
        db = records.Database('sqlite:///../data/train.db')
        conn = db.get_connection()
        query, result = table.execute_query(conn, query)
        conn.close()
        print(query, result)

    @staticmethod
    def tokenize_document(doc, print_token = False):
        operators = {'=' : 'EQL', '>' : 'GT', '<' : 'LT'}
        syntax_tokens = ["SELECT", "COUNT", "WHERE", "AND", "OR", "FROM"]
        tokens = nltk.word_tokenize(doc)
        for i in range(len(tokens)):
            if tokens[i] in syntax_tokens:
                continue
            if tokens[i] in operators.keys():
                tokens[i] = operators[tokens[i]]
            else:
                tokens[i] = tokens[i].lower()
        return tokens

    @staticmethod
    def save_dataframe(data, filename):
        data.to_json(filename, orient='records', lines=True)


    def build_tokenized_dataset(self, dataset):
        """Reads the input training files and generates a new file containing plain text sql queries"""
        self.build_table_mapping(dataset)
        queries = pd.read_json("data/" + dataset + ".jsonl", lines=True)

        count = 0
        stop_limit = len(queries)
        data = pd.DataFrame()

        # stop_limit = 10
        # iterate over the queries and convert each one to plain text sql
        for index, line in queries.iterrows():
            count += 1
            # get table and query representations
            table, query = self.get_query_from_json(line)
            
            # append query to dict, to add it to the datafram
            query_str = table.query_str(query)

            # Tokenize the query
            tokenized_query = self.tokenize_document(query_str, print_token = True)
            line["tokenized_query"] = tokenized_query
            
            # Fix the formatting of the query (lowercase + Uppercase syntax tokens)
            line["query"] = " ".join(tokenized_query)
            
            # Tokenize the question
            tokenized_question = self.tokenize_document(line["question"])
            line["tokenized_question"] = tokenized_question

            line_df = pd.DataFrame(line)
            line_df = line_df.transpose()
            data = data.append(line_df)

            if count > stop_limit:
                break
        
        # Save dataframe to file
        self.save_dataframe(data, "data/tokenized_" + dataset + ".jsonl")
