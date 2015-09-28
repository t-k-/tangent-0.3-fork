from collections import namedtuple

__author__ = 'Nidhin'


class Stats(object):
    __slots__ = ["num_documents", "num_expressions", 'missing_tags', 'problem_files']

    def __init__(self):
        self.num_documents=0
        self.num_expressions=0
        self.missing_tags={}
        self.problem_files={}

    def add(self, other):
        """
        Merge other Stats into given stats

        param other: additional stats to incorporate
        type  other: Stats
        """

        def merge_dicts(d1, d2):
            """
            Merge two dictionaries

            param d1: dictionary changed in place to have combined values
            type  d1: dictionary(key -> set)
            param d2: dictioanry to be merged
            type  d2: dictionary(key -> set)
            """
            for key,value in d2.items():
                if key not in d1:
                    d1[key] = value
                else:
                    d1[key] |= value
        
        self.num_documents += other.num_documents
        self.num_expressions += other.num_expressions
        merge_dicts(self.missing_tags, other.missing_tags)
        merge_dicts(self.problem_files, other.problem_files)

    def dump(self):
        """
        Print contents of stats
        """
        print("Total number of documents/queries processed: "+str(self.num_documents))
        print("Total number of expressions involved: "+str(self.num_expressions))
        if len(self.missing_tags) == 0:
            print("No unrecognized tags found in expressions")
        else:
            print("Unrecognized tags found in expressions:")
            for key,value in self.missing_tags.items():
                print(" ",key,": ",value)
        if len(self.problem_files) == 0:
            print("All files/queries parsed successfully")
        else:
            print("Problem files/queries:")
            for key,value in self.problem_files.items():
                print(" ",key,": ",value)
