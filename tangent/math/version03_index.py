import sys
import os
import csv

MAX_UNCOMMITED = 1000
max_results = 1000
max_size_pairs=10000

__author__ = 'FWTompa'

# modified MySQLIndexNtcir (FWT)

class Version03Index:
    def __init__(self, db, ranker=None, window=None, port=3306, host='127.0.0.1', process_id="", writer=None):
        self.ranker = ranker
        self.db = db
        self.process_id = process_id
        self.window = window
        # check for directory
        self.directory = os.path.join(os.getcwd(), "db-index")
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

    def openDB(self,fileid,topk):
        """
        Start a file for collecting query tuples to pass to search engine

        param fileid: process id used to distinguish files
        type  fileid: string
        param topk: (maximum) number of matches to return
        type  topk: int
        """
        filename = "%s_q_%s.tsv" % (self.db, fileid)
        file_path = os.path.join(self.directory, filename)
        with open(file_path, mode='w', encoding='utf-8', newline='') as file:
            writer = csv.writer(file, delimiter='\t', lineterminator='\n', quoting=csv.QUOTE_NONE, escapechar="\\")
            writer.writerow(["K",topk])
            writer.writerow(["W",self.window if self.window else 0])

    def closeDB(self,fileid,mode="q"):
        """
        Terminate a file for collecting query tuples to pass to search engine

        param fileid: process id used to distinguish files
        type  fileid: string
        param mode: "q" for querying or "i" for indexing
        type  topk: string
        """
        filename = "%s_%s_%s.tsv" % (self.db, mode, fileid)
        file_path = os.path.join(self.directory, filename)
        with open(file_path, mode='a', encoding='utf-8', newline='') as file:
            writer = csv.writer(file, delimiter='\t', lineterminator='\n', quoting=csv.QUOTE_NONE, escapechar="\\")
            writer.writerow([])
            writer.writerow(["X"])

    def add(self, expression_objects):
        """
        Add expression to index by writing into tsv file

        :param expression_objects: collection of tuples for indexing
        :type  expression_objects: list(pair(SymbolTree,list(tuples)))
        :return full fileid used to save data
        :rtype  string

        W       size
        D	docID
        E	expression	positions
        ...
        E	expression	positions
        ...
        D	docID
        ...
        X

        (but X written by CloseDB)

        N.B. tuples generated from expressions within C++ module
        """
        fileid = os.getpid()
        filename = "%s_i_%s.tsv" % (self.db, fileid)
        file_path = os.path.join(self.directory, filename)
        new = not os.path.exists(file_path) # starting a new file
        with open(file_path, mode='a', encoding='utf-8', newline='') as file:
            writer = csv.writer(file, delimiter='\t', lineterminator='\n', quoting=csv.QUOTE_NONE, escapechar="\\")
            if new:
                writer.writerow(["W",self.window if self.window else 0])
            docid = None
            for tree in expression_objects:
                if tree is not None:
                    if not docid == tree.document:
                        docid = tree.document
                        writer.writerow([])
                        writer.writerow(["D",docid])
                    expr = tree.tostring()
                    if expr != "":
                        writer.writerow(["E",expr,tree.position])
                    # tuples for pairs will be generated from expressions by C++ module
##                    pairs = tree.get_pairs(self.window)
##                    for pair in pairs: 
##                        lp, rp, rel, loc = pair.split('\t')
##                        writer.writerow(["T",lp,rp,rel,loc])
        return(fileid)

    def search(self, fileid, query_id, trees):
        """
        prepare query tuples for all trees in the query

        :param fileid: process id used to distinguish files
        :type  fileid: string
        :param query_id: canonical representation of the query (dumped SymbolTree)
        :type  query_id: string
        :param trees: collection of trees included in query
        :type  trees: list(SymbolTree)

        K       top-k
        W       window-size
        Q	queryID
        E	expression	positions
        ...
        E	expression positions
        ...
        Q	queryID
        ...
        X

        (but K written by OpenDB and X written by CloseDB)
        N.B. tuples generated from expressions within C++ module
        """

        filename = "%s_q_%s.tsv" % (self.db, fileid)
        file_path = os.path.join(self.directory, filename)
        with open(file_path, mode='a', encoding='utf-8', newline='') as file:
            writer = csv.writer(file, delimiter='\t', lineterminator='\n', quoting=csv.QUOTE_NONE, escapechar="\\")
            writer.writerow([])
            writer.writerow(["Q",query_id])
            for tree in trees:
                # get query pairs and paths
                print("search for " + tree.tostring())
                expr = tree.tostring()
                if expr != "":
                    writer.writerow(["E",expr,tree.position])
                # tuples for pairs will be generated from expressions by C++ module
##                pairs = tree.get_pairs(self.window)
##                for pair in pairs: 
##                    lp, rp, rel, loc = pair.split('\t')
##                    writer.writerow(["T",lp,rp,rel,loc])

    def get(self, fileid):
        """
        ingest result tuples for topk responses to queries

        :param fileid: process id used to distinguish files
        :type  fileid: string
        :result : list of results 
        :rtype  : dict(queryid -> list(pair(expression,score)))

        Q	queryID
        E       search-expr
        R	docID   position	expression	score
        R	docID   position	expression	score
        ...
        Q	queryID
        ...
        X

        """

        tuples = {}
        filename = "%s_r_%s.tsv" % (self.db, fileid)
        file_path = os.path.join(self.directory, filename)
        with open(file_path, mode='r', encoding='utf-8', newline='') as file:
            reader = csv.reader(file, delimiter='\t', lineterminator='\n', quoting=csv.QUOTE_NONE, escapechar="\\")
            queryid = None
            for row in reader:
                if row[0] == "Q":
                    if queryid:
                        tuples[queryid] = responses
                    queryid = row[1]
                    responses = []
                elif row[0] == "R":
                    responses.append(row[1:])
                elif row[0] == "X":
                    if queryid:
                        tuples[queryid] = responses
                        queryid = None
                else:
                    print("Invalid tuple in search response: "+row)
        if queryid:
            tuples[queryid] = responses
        return tuples
