__author__ = 'mauricio'
import os
import codecs
import sys
import time
import math
from multiprocessing import Pool

from tangent.ranking.query import Query
from tangent.ranking.ranking_functions import *
from tangent.math.symboltree import SymbolTree

CHUNK_SIZE = 2000
MAX_BATCH_SIZE=200000

DEBUG_START=0

if sys.stdout.encoding != 'utf8':
    sys.stdout = codecs.getwriter('utf8')(sys.stdout.buffer, 'strict')
if sys.stderr.encoding != 'utf8':
    sys.stderr = codecs.getwriter('utf8')(sys.stderr.buffer, 'strict')

def split_chunks(data, size, query):
    for n in range(0, len(data), size):
        yield (query, n,  data[n:n + size])

def eval_similarity(query_data):
    # do actually evaluate similarity ....
    query, start_idx, expressions = query_data

    end_idx = start_idx + len(expressions) - 1

    #create query slt
    query_name, query_expression = query
    query_tree = SymbolTree.parse_from_slt(query_expression)
    query_constraints = Query.create_default_constraints(query_tree)

    results = []
    for idx, expression_info in enumerate(expressions):
        parts = expression_info.strip().split("\t")
        expression = parts[0]
        doc_id = parts[1]
        location = parts[2]

        candidate_tree = SymbolTree.parse_from_slt(expression)

        try:
            scores, matched_q, matched_c, unified_c = similarity_v04(query_tree, candidate_tree, query_constraints)
        except:
            print("Error processing: ")
            print(query_expression, flush=True)
            print(expression, flush=True)
            print("Doc: " + doc_id, flush=True)
            print("Loc: " + location, flush=True)
            continue

        # the index is only returned because some expressions might be absent in case of errors
        results.append((scores, start_idx + idx))

    print("Processed: " + str(start_idx) + " to " + str(end_idx) + " finished", flush=True)

    return results

def read_queries(filename):
    # read all text ...
    input_file = open(filename, "r", encoding="'UTF-8'")
    all_text = input_file.readlines()
    input_file.close()

    current_query = None
    found_queries = []
    for line in all_text:
        if line[0] == "Q":
            parts = line.strip().split("\t")
            current_query = parts[1]
        if line[0] == "E":
            if current_query is not None:
                parts = line.strip().split("\t")
                current_expression = parts[1]

                found_queries.append((current_query, current_expression))

            current_query = None

    return found_queries

def join_lists(lists):
    result = []
    for list in lists:
        result += list

    return result

def save_temporal(prefix, number, results, offset):
    out_file = open(prefix + str(number) + ".tsv", "w", encoding="'UTF-8'")
    for scores, exp_idx in results:
        line = str(exp_idx + offset) + "\t" + ("\t".join([str(score) for score in scores])) + "\n"
        out_file.write(line)

    out_file.close()

def load_temporal(prefix, number):
    results = []

    in_file = open(prefix + str(number) + ".tsv", "r", encoding="'UTF-8'")
    all_lines = in_file.readlines()
    in_file.close()

    for line in all_lines:
        parts = line.strip().split("\t")

        idx = int(parts[0])
        scores = [float(part) for part in parts[1:]]

        results.append((scores, idx))

    return results

def main():
    if len(sys.argv) < 4:
        print("Usage")
        print("\tpython gold_standard_scores.py expressions output n_jobs [queries]")
        print("")
        print("Where:")
        print("\texpressions\t: File that contains all unique expressions")
        print("\toutput\t: Directory where gold standard scores will be stored")
        print("\tn_jobs\t: Number of process to use")
        print("\tqueries\t: Files that contains indexed queries")
        return

    exp_filename = sys.argv[1]
    out_prefix = sys.argv[2] + "/"

    try:
        n_jobs = int(sys.argv[3])
    except:
        print("Invalid number of jobs")
        return

    print("Using " + str(n_jobs) + " jobs", flush=True)

    exp_file = open(exp_filename, "r", encoding="'UTF-8'")
    total_bytes_input = exp_file.seek(0, os.SEEK_END)
    exp_file.close()

    all_queries = []
    for filename in sys.argv[4:]:
        all_queries += read_queries(filename)

    # sort by name
    all_queries = sorted(all_queries)


    print("Total Queries Found: " + str(len(all_queries)), flush=True)

    start_time = time.time()
    if n_jobs > 1:
        pool = Pool(processes=n_jobs)

    processed = 0
    for query_name, query_exp in all_queries:
        print("Processing: " + query_name, flush=True)

        current_query = (query_name, query_exp)

        # first create separate files containing raw scores ...
        n_partitions = 0
        last_position = 0
        first_idx = 0
        input_complete = False
        while not input_complete:
            # read next batch ...
            current_expressions = []

            exp_file = open(exp_filename, "r", encoding="'UTF-8'")
            exp_file.seek(last_position, 0)

            while len(current_expressions) < MAX_BATCH_SIZE:
                line = exp_file.readline()
                if line:
                    #add to the current batch ...
                    current_expressions.append(line)
                else:
                    # EOF reached ...
                    input_complete = True
                    break

            last_position = exp_file.tell()
            exp_file.close()

            # process batch ...
            if len(current_expressions) > 0:
                n_partitions += 1
                print("Partition " + str(n_partitions) + " (" + str(last_position) + " of " + str(total_bytes_input) + ")" , flush=True)

                if processed == 0 and n_partitions < DEBUG_START:
                    print("Skipping .... ", flush=True)
                    continue

                # execute in parallel....
                chunks = split_chunks(current_expressions, CHUNK_SIZE, current_query)

                if n_jobs > 1:
                    results = pool.map(eval_similarity, chunks)
                else:
                    # debug: Single threaded ...
                    results = []
                    for chunk in chunks:
                        results.append(eval_similarity(chunk))

                # join results ...
                results = join_lists(results)

                # save them to temporal file
                save_temporal("gs_tempo_", n_partitions - 1, results, first_idx)

                first_idx += len(current_expressions)



        # "bubble-sort" the files raw score files in pairs
        print(" ... now sorting data in temporal files ... ", flush=True)
        for idx in range(n_partitions - 1):
            # load current temporal ...
            first_part = load_temporal("gs_tempo_", idx)
            n_first = len(first_part)

            # for every other part ...
            for idx2 in range(idx + 1, n_partitions):
                second_part = load_temporal("gs_tempo_", idx2)

                # combine and sort ....
                combined = first_part + second_part
                combined = sorted(combined, reverse=True)

                # now split again ...
                first_part = combined[:n_first]
                second_part = combined[n_first:]

                # save second part to file ...
                save_temporal("gs_tempo_", idx2, second_part, 0)

            # save current part to file ...
            save_temporal("gs_tempo_", idx, first_part, 0)

        # load each split and compute the ranks while sending results to final file
        print(" ... Saving into single result file ... ", flush=True)


        # read all expressions ...
        exp_file = open(exp_filename, "r", encoding="'UTF-8'")
        all_expressions = exp_file.readlines()
        exp_file.close()

        out_file = open(out_prefix + query_name + ".tsv", "w", encoding="'UTF-8'")
        out_file.write("SLT\trank\tscore1\tscore2\tscore3\n")

        current_rank = 0
        current_scores = [-1.0, -1.0, -1.0]

        for part_idx in range(n_partitions):
            results = load_temporal("gs_tempo_", part_idx)

            for scores, exp_idx in results:
                # check if expression is no longer relevant ...
                if scores[0] <= 0.0:
                    break

                slt = all_expressions[exp_idx]
                line = slt.strip().split("\t")[0]

                for idx in range(3):
                    if scores[idx] != current_scores[idx]:
                        current_rank += 1
                        current_scores = scores

                line += "\t" + str(current_rank) + "\t" + ("\t".join([str(score) for score in scores])) + "\n"

                out_file.write(line)

            # delete temporal files
            os.remove("gs_tempo_" + str(part_idx) + ".tsv")

        out_file.close()
        processed += 1

    end_time = time.time()

    elapsed_time = end_time - start_time
    print("Total elapsed time: " + str(elapsed_time))


if __name__ == '__main__':
    main()