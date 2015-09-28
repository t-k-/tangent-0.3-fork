__author__ = 'KMDC'

import math

from tangent.math.symboltree import SymbolTree
from tangent.math.mathsymbol import MathSymbol
from tangent.math.math_extractor import MathExtractor
from tangent.math.mathml import MathML
from tangent.ranking.alignment import Alignment

def compute_matches(pairs_a, pairs_b):
    pairs_a_hashed = {}

    # create a temporary index of pairs from list a...
    for current_pair in pairs_a:
        ancestor, descendant, relation, location = current_pair

        pair_id = "[" + ancestor + "][" + descendant + "][" + relation + "]"

        if pair_id in pairs_a_hashed:
            pairs_a_hashed[pair_id].append(current_pair)
        else:
            pairs_a_hashed[pair_id] = [current_pair]

    # count matches from b
    actual_matches = []
    for pair_b in pairs_b:
        ancestor, descendant, relation, location = pair_b
        pair_id = "[" + ancestor + "][" + descendant + "][" + relation + "]"

        if pair_id in pairs_a_hashed:
            # get next matching pair
            pair_a = pairs_a_hashed[pair_id][0]
            del pairs_a_hashed[pair_id][0]

            if len(pairs_a_hashed[pair_id]) == 0:
                del pairs_a_hashed[pair_id]

            actual_matches.append((pair_a, pair_b))

    return actual_matches


# count unique elements
def count_unique_elements(pairs_list):
    elements = {}
    for ancestor, descendant, rel_location, abs_location in pairs_list:
        if ancestor not in elements:
            elements[ancestor] = {}

        if abs_location not in elements[ancestor]:
            elements[ancestor][abs_location] = 1
        else:
            elements[ancestor][abs_location] += 1

    unique = {}
    for ancestor in elements:
        unique[ancestor] = len(elements[ancestor])

    return unique


def find_common_elements(elements_a, elements_b):
    common = {}
    for element in elements_a:
        if element in elements_b:
            common[element] = min(elements_a[element], elements_b[element])

    return common


def find_pairs_with_ancestor(pairs, symbol):
    sub_pairs = []
    for ancestor, descendant, relation, location in pairs:
        if ancestor == symbol:
            sub_pairs.append((ancestor, descendant, relation, location))

    return sub_pairs


def pairs_per_element(pairs):
    per_element = {}
    for ancestor, descendant, relation, location in pairs:
        if ancestor not in per_element:
            per_element[ancestor] = []

        per_element[ancestor].append((ancestor, descendant, relation, location))

    return per_element


def pairs_per_instance(pairs):
    per_instance = {}

    for ancestor, descendant, relation, location in pairs:
        if location not in per_instance:
            per_instance[location] = []

        per_instance[location].append((ancestor, descendant, relation, location))

    return per_instance


def compute_total_unique(unique_elements):
    total_count = 0
    for ancestor in unique_elements:
        total_count += unique_elements[ancestor]

    return total_count


def list_tree_element(tree_root, path):
    current_elements = [(tree_root, path)]
    if not tree_root.next is None:
        current_elements += list_tree_element(tree_root.next, path + 'n')
    if not tree_root.above is None:
        current_elements += list_tree_element(tree_root.above, path + 'a')
    if not tree_root.below is None:
        current_elements += list_tree_element(tree_root.below, path + 'b')
    if not tree_root.pre_above is None:
        current_elements += list_tree_element(tree_root.pre_above, path + 'c')
    if not tree_root.pre_below is None:
        current_elements += list_tree_element(tree_root.pre_below, path + 'd')
    if not tree_root.within is None:
        current_elements += list_tree_element(tree_root.within, path + 'w')
    if not tree_root.element is None:
        current_elements += list_tree_element(tree_root.element, path + 'e')

    return current_elements


def get_fmeasure(matches, size_query, size_candidate):
    if int(matches) == 0 or int(size_query) == 0 or int(size_candidate) == 0:
        return 0.0

    recall = matches / float(size_query)
    precision = matches / float(size_candidate)

    fmeasure = (2.0 * recall * precision) / (recall + precision)

    return fmeasure


def get_child_path(parent_loc, child_loc):
    if parent_loc == "-" or parent_loc == "":
        extended = ""
    elif "0" <= parent_loc[0] <= "9":
        extended = MathSymbol.rldecode(parent_loc)
    else:
        extended = parent_loc

    if len(child_loc) > 0 and "0" <= child_loc[0] <= "9":
        extended += MathSymbol.rldecode(child_loc)
    else:
        extended += child_loc

    if len(extended) > 5:
        return MathSymbol.rlencode(extended)
    elif len(extended) > 0:
        return extended
    else:
        return "-"


# Original f-measure of matched pairs...
def similarity_v00(pairs_query, pairs_candidate):
    if len(pairs_query) == 0 or len(pairs_candidate) == 0:
        return [0.0], {}, {}

    matches = compute_matches(pairs_query, pairs_candidate)

    n_matches = float(len(matches))
    fmeasure = get_fmeasure(n_matches, len(pairs_query), len(pairs_candidate))

    highlight_a = {}
    highlight_b = {}
    for pair_a, pair_b in matches:
        # from a ...
        ancestor, descendant, relation, location = pair_a
        if location not in highlight_a:
            highlight_a[location] = ancestor
        child_location = get_child_path(location, relation)
        if child_location not in highlight_a:
            highlight_a[child_location] = descendant

        # from b ...
        ancestor, descendant, relation, location = pair_b
        if location not in highlight_b:
            highlight_b[location] = ancestor
        child_location = get_child_path(location, relation)
        if child_location not in highlight_b:
            highlight_b[child_location] = descendant

    # print(pairs_candidate)
    #print(matches)
    #print(highlight_b)
    #print("")

    return [fmeasure], highlight_a, highlight_b


def align_trees(root_1, path_1, root_2, path_2, size_query):
    matches = []
    if root_1.tag == root_2.tag:
        matches.append(Alignment(root_1, path_1, root_2, path_2))

    root_fmeasure = get_fmeasure(len(matches), size_query, 1)

    total_unmatched = 1 - len(matches)
    total_elements = 1

    # children to be tested for alignment....
    children_tests = []
    if root_1.next is not None and root_2.next is not None:
        children_tests.append((root_1.next, root_2.next, 'n'))
    if root_1.above is not None and root_2.above is not None:
        children_tests.append((root_1.above, root_2.above, 'a'))
    if root_1.below is not None and root_2.below is not None:
        children_tests.append((root_1.below, root_2.below, 'b'))
    if root_1.pre_above is not None and root_2.pre_above is not None:
        children_tests.append((root_1.pre_above, root_2.pre_above, 'c'))
    if root_1.pre_below is not None and root_2.pre_below is not None:
        children_tests.append((root_1.pre_below, root_2.pre_below, 'd'))
    if root_1.within is not None and root_2.within is not None:
        children_tests.append((root_1.within, root_2.within, 'w'))
    if root_1.element is not None and root_2.element is not None:
        children_tests.append((root_1.element, root_2.element, 'e'))

    for child_1, child_2, relation in children_tests:
        child_matched, child_unmatched, child_score = align_trees(child_1, path_1 + relation,
                                                                  child_2, path_2 + relation, size_query)
        child_total = len(child_matched) + child_unmatched

        child_fmeasure = get_fmeasure(len(matches) + len(child_matched), size_query, total_elements + child_total)
        if child_fmeasure > root_fmeasure:
            matches += child_matched
            total_unmatched += child_unmatched
            total_elements += child_total

            root_fmeasure = child_fmeasure

    return matches, total_unmatched, root_fmeasure


def test_alignment(alignment, p_alignments, size_query):
    # use pairs to compute score (and subtree)
    matches, unmatched, score = align_trees(alignment.q_element, alignment.q_location,
                                            alignment.c_element, alignment.c_location, size_query)

    # remove matches from list of alignments...
    for alignment in matches:
        p_alignments.remove(alignment)

    return matches, unmatched, score


def similarity_score_from_alignments(tree_query, tree_candidate, accepted_matches):
    # initially, use only the score from the top match alone...
    highlight_a = {}
    highlight_b = {}
    if len(accepted_matches) > 0:
        score, alignment, matches = accepted_matches[0]
        for match in matches:
            loc = get_child_path(match.q_location, "")
            if loc not in highlight_a:
                highlight_a[loc] = match.q_element

            loc = get_child_path(match.c_location, "")
            if loc not in highlight_b:
                highlight_b[loc] = match.c_element

    else:
        score = 0.0

    return score, highlight_a, highlight_b


# full alignment greedy function...
def similarity_v01(tree_query, tree_candidate):
    # compute all candidate alignments
    all_alignments = []

    all_nodes_query = list_tree_element(tree_query.root, '')
    all_nodes_candidate = list_tree_element(tree_candidate.root, '')

    for elem_q, loc_q in all_nodes_query:
        for elem_c, loc_c in all_nodes_candidate:

            # Pending to adapt to wildcards...
            # if u == v or u == wildcard
            if elem_q.tag == elem_c.tag:
                alignment = Alignment(elem_q, loc_q, elem_c, loc_c)
                all_alignments.append(alignment)

    # Now, compute a score for each possible sub alignment
    p_alignments = list(all_alignments)
    query_size = len(all_nodes_query)
    scored_alignments = []

    for alignment in all_alignments:
        if alignment in p_alignments:
            matches, unmatched, score = test_alignment(alignment, p_alignments, query_size)

            # print(str(len(matches)) + ", " + str(unmatched))
            scored_alignments.append((score, alignment, matches))

    scored_alignments = sorted(scored_alignments, reverse=True, key=lambda x: x[0])

    aligned_candidate = {}
    aligned_query = {}
    final_alignments = []
    for score, alignment, matches in scored_alignments:
        if alignment.q_location not in aligned_query and alignment.c_location not in aligned_candidate:

            # root not yet aligned, align it with all matched elements!
            for match_alignment in matches:
                aligned_query[match_alignment.q_location] = True
                aligned_candidate[match_alignment.c_location] = True

            # add the accepted alignment ....
            final_alignments.append((score, alignment, matches))

    # print("Computed: " + str(len(scored_alignments)) + ", accepted: " + str(len(final_alignments)))
    score, highlight_a, highlight_b = similarity_score_from_alignments(tree_query, tree_candidate, final_alignments)

    return [score], highlight_a, highlight_b


# greedy similarity function
def similarity_v02(pairs_query, pairs_candidate):
    # Find the unique elements and their counts on each expression
    e_query = count_unique_elements(pairs_query)
    e_candidate = count_unique_elements(pairs_candidate)

    # Compute counts of shared elements
    overlap = find_common_elements(e_query, e_candidate)

    # separate pairs....
    pairs_element_query = pairs_per_element(pairs_query)
    pairs_element_candidate = pairs_per_element(pairs_candidate)

    # Get a subset of pairs from the candidate which comes
    # from a limited number of overlapping unique elements
    final_pairs_candidate = []

    total_elements_matched = 0
    for ancestor in overlap:
        count = overlap[ancestor]

        total_elements_matched += count

        # Get all pairs for the current symbol from candidate
        sub_candidate_pairs = pairs_element_candidate[ancestor]

        if count < e_candidate[ancestor]:
            # Get all pairs for that symbol from query
            sub_query_pairs = pairs_element_query[ancestor]

            # Organize candidate pairs by instance....
            sub_pairs_per_instance = pairs_per_instance(sub_candidate_pairs)

            scored_list = []
            for location in sub_pairs_per_instance:
                matches = compute_matches(sub_query_pairs, sub_pairs_per_instance[location])
                score = len(matches)
                scored_list.append((score, location))

            sorted_list = sorted(scored_list, reverse=True)

            for i in range(count):
                score, location = sorted_list[i]

                # add pairs...
                final_pairs_candidate += sub_pairs_per_instance[location]
        else:
            # Add all pairs....
            final_pairs_candidate += sub_candidate_pairs

    # return Compute_FMeasure(pairs_query, final_pairs_candidate)
    scores, highlight_a, highlight_b = similarity_v00(pairs_query, final_pairs_candidate)
    pair_fmeasure = scores[0]

    if total_elements_matched > 0:
        elements_recall = float(total_elements_matched) / compute_total_unique(e_query)
        elements_precision = float(total_elements_matched) / compute_total_unique(e_candidate)

        elements_fmeasure = (2.0 * elements_recall * elements_precision) / (elements_precision + elements_recall)
    else:
        elements_fmeasure = 0.0

    return [pair_fmeasure, elements_fmeasure], highlight_a, highlight_b


def check_tag_is_variable(tag):
    return tag[0:2] == "V!" or tag[0] == "?"


def check_tag_is_number(tag):
    return tag[0:2] == "N!"

def check_tag_is_matrix(tag):
    return tag[0:2] == "M!"

def check_tag_has_type(tag):
    return tag[1:2] == "!"

def generate_unification_pairs(tag_pairs):
    # This function assumes that all pairs in the input have a common ancestor ...
    unification_pairs = []

    for ancestor, descendant, relation, location in tag_pairs:
        if descendant == ancestor:
            u_descendant = "<U>"
        elif check_tag_is_variable(descendant):
            u_descendant = "<V>"
        else:
            u_descendant = descendant

        unification_pairs.append(("<U>", u_descendant, relation, location))

    return unification_pairs


def unify_variables(pairs, variables):
    new_pairs = []

    for ancestor, descendent, relation, location in pairs:
        # check if variable...
        if ancestor in variables:
            if variables[ancestor] is not None:
                ancestor = "U!" + str(variables[ancestor])
        else:
            # check if constant
            if ancestor[0:2] == "N!":
                # single value for all constants
                ancestor = "N!U"

        if descendent in variables:
            if variables[descendent] is not None:
                descendent = "U!" + str(variables[descendent])
        else:
            # check if constant
            if descendent[0:2] == "N!":
                # single value for all constants
                descendent = "N!U"

        new_pairs.append((ancestor, descendent, relation, location))

    return new_pairs


# Greedy pair matching with rough alignments and greedy unification
def similarity_v03(pairs_query, pairs_candidate):
    # Find the unique elements and their counts on each expression
    e_query = count_unique_elements(pairs_query)
    e_candidate = count_unique_elements(pairs_candidate)

    # separate pairs....
    pairs_element_query = pairs_per_element(pairs_query)
    pairs_element_candidate = pairs_per_element(pairs_candidate)

    # identify all variables...
    v_query = {}
    v_candidate = {}

    # ... on query...
    unification_pairs_query = {}
    unification_pairs_candidate = {}
    for tag in e_query:
        if check_tag_is_variable(tag):
            v_query[tag] = None
            unification_pairs_query[tag] = generate_unification_pairs(pairs_element_query[tag])

    # ... on candidate ...
    for tag in e_candidate:
        if check_tag_is_variable(tag):
            v_candidate[tag] = None
            unification_pairs_candidate[tag] = generate_unification_pairs(pairs_element_candidate[tag])

    # Evaluate all possible unifications ...
    unifications = []
    unification_weights = []
    for query_var in v_query:
        for candidate_var in v_candidate:
            # test unification between query_var and candidate_var
            matches = compute_matches(unification_pairs_query[query_var], unification_pairs_candidate[candidate_var])
            fmeasure = get_fmeasure(len(matches),
                                    len(unification_pairs_query[query_var]),
                                    len(unification_pairs_candidate[candidate_var]))
            extra_score = 1.0 if query_var == candidate_var else 0.0

            weight = ((fmeasure, extra_score), query_var, candidate_var)
            unification_weights.append(weight)


    # greedily accept unifications with the most matching pairs ...
    unification_weights = sorted(unification_weights, reverse=True)
    for scores, query_var, candidate_var in unification_weights:
        fmeasure, equal = scores
        if fmeasure > 0.0:
            # check if variables have not been unified yet...
            if v_query[query_var] is None and v_candidate[candidate_var] is None:
                # Accept unification...
                u_idx = len(unifications)
                unifications.append((query_var, candidate_var))
                v_query[query_var] = u_idx
                v_candidate[candidate_var] = u_idx

    # print(unifications)
    unified_pairs_query = unify_variables(pairs_query, v_query)
    unified_pairs_candidate = unify_variables(pairs_candidate, v_candidate)

    # compute similarity (with unification)
    u_scores, u_highlight_q, u_highlight_c = similarity_v02(unified_pairs_query, unified_pairs_candidate)
    # compute similarity (without unification)
    o_scores, o_highlight_c, o_highlight_c = similarity_v02(pairs_query, pairs_candidate)

    unified_pair_fmeasure = u_scores[0]
    unified_cc_fmeasure = u_scores[1]
    original_pair_fmeasure = o_scores[0]

    scores = [unified_pair_fmeasure, original_pair_fmeasure, unified_cc_fmeasure]

    # compute unified locations ...
    unified_c = []
    for location in u_highlight_c:
        if location not in o_highlight_c:
            unified_c.append(location)

    return scores, u_highlight_q, u_highlight_c, unified_c

def get_matrix_size(matrix_tag):
    size_middle = matrix_tag.find("x")

    if size_middle == -1:
        # invalid tag!
        return (-1, -1)
    else:
        cols = int(matrix_tag[size_middle + 1:])
        start = size_middle - 1
        while start > 1 and "0" <= matrix_tag[start - 1] <= "9":
            start -= 1
        rows = int(matrix_tag[start:size_middle])

        return (rows, cols)

def get_element_children(root):
    children = []

    if root.within:
        children.append(root.within)

        while children[-1].element:
            children.append(children[-1].element)

    return children

def align_trees_unification(root_1, path_1, root_2, path_2, root_c, restricted_vars, query_size):
    matched = []
    unifiable_qvars = []
    unifiable_vars = []
    unifiable_const = []
    total_unmatched = 0

    r1_is_var = check_tag_is_variable(root_1.tag)
    r2_is_var = check_tag_is_variable(root_2.tag)

    current_alignment = Alignment(root_1, path_1, root_2, path_2)

    # check current alignment ...
    if root_1.tag[0] == "?":
        # it's a query variable, it could be align with almost anything ...
        if root_c.tag.check_unifiable(root_1, root_2):
            unifiable_qvars.append(current_alignment)
        else:
            total_unmatched += 1

    elif r1_is_var and r2_is_var:
        # check if constraints var ....
        if (root_1.tag in restricted_vars) or (root_2.tag in restricted_vars):
            # can only be matched exactly
            if root_1.tag == root_2.tag:
                matched.append(current_alignment)
            else:
                total_unmatched += 1
        else:
            # all other variables will be considered unifiable (even when exact matches)
            unifiable_vars.append(current_alignment)
    elif root_1.tag == root_2.tag:
        # exact match
        matched.append(current_alignment)
    else:
        # check if unifiable
        if root_c.tag.check_unifiable(root_1, root_2):
            # constraints allowed unification ...
            unifiable_const.append(current_alignment)
        else:
            # completely unmatched...
            total_unmatched += 1

    num_max_matches = len(matched) + len(unifiable_vars) + len(unifiable_const)
    current_size = 1
    root_max_fmeasure = get_fmeasure(num_max_matches, query_size, current_size)

    # children to be tested for alignment....
    children_tests = []
    if root_1.next is not None and root_2.next is not None:
        children_tests.append((root_1.next, root_2.next, root_c.next, 'n', 'n'))
    if root_1.above is not None and root_2.above is not None:
        children_tests.append((root_1.above, root_2.above, root_c.above, 'a', 'a'))
    if root_1.below is not None and root_2.below is not None:
        children_tests.append((root_1.below, root_2.below, root_c.below, 'b', 'b'))
    if root_1.pre_above is not None and root_2.pre_above is not None:
        children_tests.append((root_1.pre_above, root_2.pre_above, root_c.pre_above, 'c', 'c'))
    if root_1.pre_below is not None and root_2.pre_below is not None:
        children_tests.append((root_1.pre_below, root_2.pre_below, root_c.pre_below, 'd', 'd'))

    if root_1.tag[0:2] == "M!" and root_2.tag[0:2] == "M!":
        m_rows_1, m_cols_1 = get_matrix_size(root_1.tag)
        m_rows_2, m_cols_2 = get_matrix_size(root_2.tag)

        children_1 = get_element_children(root_1)
        children_c = get_element_children(root_c)
        children_2 = get_element_children(root_2)

        if (m_rows_1 == 1 or m_cols_1 == 1) and (m_rows_2 == 1 or m_cols_2 == 1):
            # both are one-dimensional, compare as a list...
            child_path = "w"
            for child_idx in range(min(len(children_1), len(children_2))):
                children_tests.append((children_1[child_idx], children_2[child_idx], children_c[child_idx],
                                       child_path, child_path))
                child_path += "e"
        else:
            # at least one is a matrix, compare as matrices
            for child_row in range(min(m_rows_1, m_rows_2)):
                for child_col in range(min(m_cols_1, m_cols_2)):
                    child_idx_1 = child_row * m_cols_1 + child_col
                    child_idx_2 = child_row * m_cols_2 + child_col

                    if child_idx_1 >= len(children_1) or child_idx_2 >= len(children_2):
                         # bad formed matrix found!
                        print("Warning: Bad matrix found")

                    child_1 = children_1[child_idx_1]
                    child_c = children_c[child_idx_1]
                    child_2 = children_2[child_idx_2]

                    child_path_1 = "w" + "e" * child_idx_1
                    child_path_2 = "w" + "e" * child_idx_2

                    children_tests.append((child_1, child_2, child_c, child_path_1, child_path_2))

        """
        if root_1.within is not None and root_2.within is not None:
            children_tests.append((root_1.within, root_2.within, root_c.within, 'w'))
        if root_1.element is not None and root_2.element is not None:
            children_tests.append((root_1.element, root_2.element, root_c.element, 'e'))
        """
    else:
        # other elements different from M! that might contain elements within
        if root_1.within is not None and root_2.within is not None:
            children_tests.append((root_1.within, root_2.within, root_c.within, 'w', 'w'))



    for child_1, child_2, constrain, relation1, relation2 in children_tests:
        child_res = align_trees_unification(child_1, path_1 + relation1, child_2, path_2 + relation2,
                                            constrain, restricted_vars, query_size)
        child_matched, child_u_qvars, child_u_vars, child_u_const, child_unmatched = child_res

        # compute max f-measure after adding child branch to current root..
        child_potential_matches = len(child_matched) + len(child_u_qvars) + len(child_u_vars) + len(child_u_const)
        child_max_matches = child_potential_matches + num_max_matches
        child_size = current_size + (child_potential_matches + child_unmatched)
        child_max_fmeasure = get_fmeasure(child_max_matches, query_size, child_size)

        if child_max_fmeasure > root_max_fmeasure:
            matched += child_matched
            unifiable_qvars += child_u_qvars
            unifiable_vars += child_u_vars
            unifiable_const += child_u_const
            total_unmatched += child_unmatched

            num_max_matches = child_max_matches
            current_size = child_size
            root_max_fmeasure = child_max_fmeasure

    return matched, unifiable_qvars, unifiable_vars, unifiable_const, total_unmatched

def tree_size_from_location(locations):
    if len(locations) > 0:
        min_len = None
        max_len = None
        full_locations = {}
        for loc in locations:
            current_len = len(loc)

            if min_len is None or current_len < min_len:
                min_len = current_len
            if max_len is None or current_len > max_len:
                max_len = current_len

            if current_len in full_locations:
                full_locations[current_len].append(loc)
            else:
                full_locations[current_len] = [loc]

        total_nodes = 0
        current_len = max_len
        while current_len >= min_len:
            local_count = len(full_locations[current_len])
            total_nodes += local_count

            if local_count >= 2 or current_len > min_len:
                # must have parents as part of the tree ...
                for loc in full_locations[current_len]:
                    prefix = loc[:-1]

                    if current_len - 1 in full_locations:
                        # already exists ... check for duplicates ...
                        if prefix not in full_locations[current_len - 1]:
                            # new parent, add
                            full_locations[current_len - 1].append(prefix)
                    else:
                        # add ...
                        full_locations[current_len - 1] = [prefix]
                        if current_len - 1 < min_len:
                            min_len = current_len - 1

            current_len -= 1

        return total_nodes
    else:
        return 0

def matched_edges_from_locations(locations):
    if len(locations) > 0:
        min_len = None
        max_len = None
        full_locations = {}
        for loc in locations:
            current_len = len(loc)

            if min_len is None or current_len < min_len:
                min_len = current_len
            if max_len is None or current_len > max_len:
                max_len = current_len

            if current_len in full_locations:
                full_locations[current_len].append(loc)
            else:
                full_locations[current_len] = [loc]

        total_edges = 0
        current_len = max_len
        while current_len > min_len:
            # check ...
            if (current_len in full_locations) and ((current_len - 1) in full_locations):
                # only do test if parents might appear
                for loc in full_locations[current_len]:
                    prefix = loc[:-1]

                    if prefix in full_locations[current_len - 1]:
                        total_edges += 1

            current_len -= 1

        return total_edges
    else:
        return 0

def greedy_unification(unifiable_alignments):
    # ideal unification method (optimization -> pair matching, Hungarian method) might be too heavy.
    # use a faster sub-optimal greedy unification instead

    # the unified tags
    q_unified = {}
    c_unified = {}

    # ... first, identify the possible unifications and count their frequency
    q_vars = {}
    for u_alignment in unifiable_alignments:
        # ... count frequency of matches between query and candidate vars
        if u_alignment.q_element.tag not in q_vars:
            q_vars[u_alignment.q_element.tag] = {}
            q_unified[u_alignment.q_element.tag] = None

        if u_alignment.c_element.tag not in q_vars[u_alignment.q_element.tag]:
            q_vars[u_alignment.q_element.tag][u_alignment.c_element.tag] = 1
        else:
            q_vars[u_alignment.q_element.tag][u_alignment.c_element.tag] += 1

        # ... identify the candidate vars
        if u_alignment.c_element.tag not in c_unified:
            c_unified[u_alignment.c_element.tag] = None

    # ... sort possible unification by frequency
    sorted_vars = []
    for q_tag in q_vars:
        for c_tag in q_vars[q_tag]:
            extra_score = 1 if q_tag == c_tag else 0
            sorted_vars.append(((q_vars[q_tag][c_tag], extra_score), q_tag, c_tag))
    sorted_vars = sorted(sorted_vars, reverse=True)

    # ... unify variables preferring most frequent first  ...
    for scores, q_tag, c_tag in sorted_vars:
        if q_unified[q_tag] is None and c_unified[c_tag] is None:
            # accept unification...
            # also, ensure 1 to 1 relation...
            q_unified[q_tag] = c_tag
            c_unified[c_tag] = q_tag

    return q_unified, c_unified

def test_alignment_unification(alignment, p_alignments, restricted_vars, size_query, candidate_size):
    # compute best structural alignment subtree
    alignment_results = align_trees_unification(alignment.q_element, alignment.q_location,
                                                alignment.c_element, alignment.c_location,
                                                alignment.constraint, restricted_vars, size_query)
    matched, unifiable_qvars, unifiable_vars, unifiable_const, total_unmatched = alignment_results

    unified = []
    not_unified = 0

    # greedy unification of query variables ...
    if len(unifiable_qvars) > 0:
        var_q_unified, var_c_unified = greedy_unification(unifiable_qvars)
        for u_alignment in unifiable_qvars:
            # check ...
            if var_q_unified[u_alignment.q_element.tag] == u_alignment.c_element.tag:
                unified.append(u_alignment)
            else:
                not_unified += 1
    else:
        var_q_unified = {}
        var_c_unified = {}

    # greedy unification of normal variables
    q_unified, c_unified = greedy_unification(unifiable_vars)

    # ... apply unified variables to compute total matches ...
    for u_alignment in unifiable_vars:
        # check ...
        if q_unified[u_alignment.q_element.tag] == u_alignment.c_element.tag:
            if u_alignment.q_element.tag == u_alignment.c_element.tag:
                # same variable, count as standard match...
                matched.append(u_alignment)
            else:
                unified.append(u_alignment)
        else:
            not_unified += 1

    # ... and count the unified constants as well
    for u_alignment in unifiable_const:
        unified.append(u_alignment)

    # Now, remove the alignments tested....
    # ... remove matches from list of alignments...
    for m_alignment in matched:
        if m_alignment in p_alignments:
            p_alignments.remove(m_alignment)

    # also, remove unified from list of alignments...
    for u_alignment in unified:
        if u_alignment in p_alignments:
            p_alignments.remove(u_alignment)

    # check case where root was a variable and was not unified ...
    if alignment in p_alignments:
        # root was not unified or matched. Remove from pending alignments anyway ...
        p_alignments.remove(alignment)

    # now, compute scores for current alignment ...
    unified_matches = len(matched) + len(unified)
    total_unmatched += not_unified

    unified_locations = [match.c_location for match in (matched + unified)]
    matched_edges = matched_edges_from_locations(unified_locations)

    u_sym_rec = unified_matches / float(size_query)
    if size_query > 1:
        if matched_edges > 0:
            u_edge_rec = matched_edges / float(size_query - 1)
        else:
            # avoid making f-measure 0 if nodes were matched but no edges were matches
            # assume that less than one edge was matched but not zero
            u_edge_rec = 0.5 / float(size_query - 1)
    else:
        # no edges to match ...
        u_edge_rec = 1.0

    if u_sym_rec + u_edge_rec > 0:
        combined_fmeasure = (2.0 * u_sym_rec * u_edge_rec) / (u_sym_rec + u_edge_rec)
    else:
        combined_fmeasure = 0.0

    global_u_unmatched = candidate_size - unified_matches

    normal_matches = len(matched)

    scores = [combined_fmeasure, -global_u_unmatched, normal_matches]

    unification = (q_unified, c_unified, var_q_unified, var_c_unified)
    match_data = (unified_matches, matched_edges, unification)

    return matched, unified, total_unmatched, scores, match_data


def highlighting_from_alignments(accepted_matches):
    highlight_q = {}
    highlight_c = {}
    unified_loc_c = {}

    # for each alignment
    for scores, alignment, matched, unified, match_data in accepted_matches:
        # now, to highlight the matches ...
        for match in matched:
            loc = get_child_path(match.q_location, "")
            highlight_q[loc] = match.q_element

            loc = get_child_path(match.c_location, "")
            highlight_c[loc] = match.c_element

        for match in unified:
            loc = get_child_path(match.c_location, "")
            unified_loc_c[loc] = match.c_element

    return highlight_q, highlight_c, unified_loc_c

# Full alignment function with greedy unification ...
def similarity_v04(tree_query, tree_candidate, tree_constraints):
    # compute all candidate alignments
    all_alignments = []

    # 1) first, list all nodes from the tres
    all_nodes_query = list_tree_element(tree_query.root, '')
    all_nodes_candidate = list_tree_element(tree_candidate.root, '')
    all_nodes_constraints = list_tree_element(tree_constraints.root, '')

    # 2) second, check all possible alignments and restricted vars
    restricted_vars = []
    for idx, q_node in enumerate(all_nodes_query):
        elem_q, loc_q = q_node
        const_info, const_loc = all_nodes_constraints[idx]

        # verify ...
        if loc_q != const_loc:
            print("Warning: Invalid constraint tree used")

        # check if variable and restricted
        if not const_info.tag.unifiable and elem_q.tag[0:2] == "V!":
            # variable not unifiable
            if not elem_q.tag in restricted_vars:
                restricted_vars.append(elem_q.tag)

        for elem_c, loc_c in all_nodes_candidate:
            if const_info.tag.check_unifiable(elem_q, elem_c):
                alignment = Alignment(elem_q, loc_q, elem_c, loc_c, const_info)
                all_alignments.append(alignment)

    # Now, compute a score for each possible sub alignment
    p_alignments = list(all_alignments)
    query_size = len(all_nodes_query)
    candidate_size = len(all_nodes_candidate)
    scored_alignments = []

    for alignment in all_alignments:
        if alignment in p_alignments:
            result = test_alignment_unification(alignment, p_alignments, restricted_vars, query_size, candidate_size)
            matches, unified, unmatched, scores, match_data = result

            scored_alignments.append((scores, alignment, matches, unified, match_data))

    scored_alignments = sorted(scored_alignments, reverse=True, key=lambda x: x[0])

    # use only top 1 alignment ...
    if len(scored_alignments) > 0:
        scores, alignment, matches, unified, match_data = scored_alignments[0]

        highlight_q, highlight_c, unified_c = highlighting_from_alignments([scored_alignments[0]])
    else:
        # no matches ...
        scores = [0.0, 0, 0]
        highlight_q = {}
        highlight_c = {}
        unified_c = {}

    return scores, highlight_q, highlight_c, unified_c

def similarity_scores_from_u_alignments(accepted_matches, query_size, candidate_size):
    # For now, use only the score from the top match alone...

    if len(accepted_matches) > 0:
        # count nodes and edges
        total_exact_matches = 0
        total_unified_matches = 0
        total_edges = 0

        for scores, alignment, matched, unified, match_data in accepted_matches:
            unified_matches, valid_edges = match_data

            # count
            total_exact_matches += len(matched)
            total_unified_matches += unified_matches
            total_edges += valid_edges

        u_sym_rec = total_unified_matches / float(query_size)
        if query_size > 1:
            if total_edges > 0:
                u_edge_rec = total_edges / float(query_size - 1)
            else:
                # avoid making f-measure 0 if nodes were matched but no edges were matches
                # assume that less than one edge was matched but not zero
                u_edge_rec = 0.5 / float(query_size - 1)
        else:
            # no edges to match ...
            u_edge_rec = 1.0

        if u_sym_rec + u_edge_rec > 0:
            combined_fmeasure = (2.0 * u_sym_rec * u_edge_rec) / (u_sym_rec + u_edge_rec)
        else:
            combined_fmeasure = 0.0

        global_u_unmatched = candidate_size - total_unified_matches

        normal_matches = total_exact_matches

        scores = [combined_fmeasure, -global_u_unmatched, normal_matches]

    else:
        scores = [0.0, 0.0, 0.0]

    return scores


# Full alignment function with greedy unification ...
def similarity_v05(tree_query, tree_candidate, tree_constraints):
    # compute all candidate alignments
    all_alignments = []

    # 1) first, list all nodes from the tres
    all_nodes_query = list_tree_element(tree_query.root, '')
    all_nodes_candidate = list_tree_element(tree_candidate.root, '')
    all_nodes_constraints = list_tree_element(tree_constraints.root, '')

    # 2) second, check all possible alignments and restricted vars
    restricted_vars = []
    for idx, q_node in enumerate(all_nodes_query):
        elem_q, loc_q = q_node
        const_info, const_loc = all_nodes_constraints[idx]

        # verify ...
        if loc_q != const_loc:
            print("Warning: Invalid constraint tree used")

        # check if variable and restricted
        if not const_info.tag.unifiable and elem_q.tag[0:2] == "V!":
            # variable not unifiable
            if not elem_q.tag in restricted_vars:
                restricted_vars.append(elem_q.tag)

        for elem_c, loc_c in all_nodes_candidate:
            if const_info.tag.check_unifiable(elem_q, elem_c):
                alignment = Alignment(elem_q, loc_q, elem_c, loc_c, const_info)
                all_alignments.append(alignment)

    # Now, compute a score for each possible sub alignment
    p_alignments = list(all_alignments)
    query_size = len(all_nodes_query)
    candidate_size = len(all_nodes_candidate)
    scored_alignments = []

    for alignment in all_alignments:
        if alignment in p_alignments:
            result = test_alignment_unification(alignment, p_alignments, restricted_vars, query_size, candidate_size)

            matches, unified, unmatched, scores, match_data = result

            scored_alignments.append((scores, alignment, matches, unified, match_data))

    scored_alignments = sorted(scored_alignments, reverse=True, key=lambda x: x[0])

    aligned_candidate = {}
    aligned_query = {}

    unified_query = {}
    unified_candidate = {}
    unified_qvar_query = {}
    unified_qvar_candidate = {}

    final_alignments = []
    for idx, alignment_data in enumerate(scored_alignments):
        scores, alignment, matches, unified, match_data  = alignment_data
        align_unified_matches, align_matched_edges, align_unification = match_data
        align_q_unified, align_c_unified, align_qvar_q_unified, align_qvar_c_unified = align_unification

        # check valid matches/unifications
        if idx == 0:
            # everything is accepted for first, largest match
            valid_matches = list(matches)
            valid_unified = list(unified)

            valid_edges = align_matched_edges

            unified_query = align_q_unified
            unified_candidate = align_c_unified

            unified_qvar_query = align_qvar_q_unified
            unified_qvar_candidate = align_qvar_c_unified
        else:
            valid_matches = []
            valid_unified = []

            # start with current unification ...
            tempo_unified_query = dict(unified_query)
            tempo_unified_candidate = dict(unified_candidate)

            tempo_unified_qvar_query = dict(unified_qvar_query)
            tempo_unified_qvar_candidate = dict(unified_qvar_candidate)

            # add any extra variable unification that is compatible with current model
            maps_changed = False
            invalid_mapping = {}
            for var_q in align_q_unified:
                var_c = align_q_unified[var_q]

                if ((var_c is not None) and
                    ((not var_c in tempo_unified_candidate) or (tempo_unified_candidate[var_c] is None)) and
                    ((not var_q in tempo_unified_query) or (tempo_unified_query[var_q] is None))):
                    # map can be updated
                    maps_changed = True
                    tempo_unified_query[var_q] = var_c
                    tempo_unified_candidate[var_c] = var_q
                else:
                    if (var_c is not None) and (var_q in tempo_unified_query) and (tempo_unified_query[var_q] != var_c):
                        invalid_mapping[var_q] = var_c

            # repeat test for qvars
            qvar_maps_changed = False
            for var_q in align_qvar_q_unified:

                var_c = align_qvar_q_unified[var_q]

                if ((var_c is not None) and
                    ((not var_c in tempo_unified_qvar_candidate) or (tempo_unified_qvar_candidate[var_c] is None)) and
                    ((not var_q in tempo_unified_qvar_query) or (tempo_unified_qvar_query[var_q] is None))):
                    # map can be updated
                    qvar_maps_changed = True
                    tempo_unified_qvar_query[var_q] = var_c
                    tempo_unified_qvar_candidate[var_c] = var_q
                else:
                    if ((var_c is not None) and
                        (var_q in tempo_unified_qvar_query) and (tempo_unified_qvar_query[var_q] != var_c)):
                        invalid_mapping[var_q] = var_c


            # now that unification has been merged, re-compute matches and unified
            # (nodes that are still matches or unifiable and that do not overlap previous matches)
            submatch_locations = []
            for match_alignment in matches:
                # check non-overlapping and valid mapping
                if ((match_alignment.q_location not in aligned_query) and
                    (match_alignment.c_location not in aligned_candidate) and
                    (match_alignment.q_element.tag not in invalid_mapping)):

                    valid_matches.append(match_alignment)
                    submatch_locations.append(match_alignment.q_location)

            for unified_alignment in unified:
                # check non-overlapping and valid mapping
                if ((unified_alignment.q_location not in aligned_query) and
                    (unified_alignment.c_location not in aligned_candidate) and
                    (unified_alignment.q_element.tag not in invalid_mapping)):

                    valid_unified.append(unified_alignment)
                    submatch_locations.append(unified_alignment.q_location)

            if len(valid_matches) + len(valid_unified) > 0:
                # additional valid matches were found ...
                # replace unification ...
                if maps_changed:
                    unified_query = tempo_unified_query
                    unified_candidate = tempo_unified_candidate

                if qvar_maps_changed:
                    unified_qvar_query = tempo_unified_qvar_query
                    unified_qvar_candidate = tempo_unified_qvar_candidate

                valid_edges = matched_edges_from_locations(submatch_locations)
            else:
                # nothing to add ...
                continue

        # mark the accepted matches ....
        for match_alignment  in valid_matches:
            aligned_query[match_alignment.q_location] = True
            aligned_candidate[match_alignment.c_location] = True

        # mark the accepted unification ....
        for unified_alignment in valid_unified:
            aligned_query[unified_alignment.q_location] = True
            aligned_candidate[unified_alignment.c_location] = True

        # add the accepted alignment ....
        match_data = (len(valid_matches) + len(valid_unified), valid_edges)
        final_alignments.append((scores, alignment, valid_matches, valid_unified, match_data))

    """
    print("Final alignments: ")
    for scores, alignment, matches, unified, match_data in final_alignments:
        print("-------")
        print(scores)
        print(unified)
        print(matches)
        print(match_data)
    """
    # print("Computed: " + str(len(scored_alignments)) + ", accepted: " + str(len(final_alignments)))

    scores = similarity_scores_from_u_alignments(final_alignments, query_size, candidate_size)
    highlight_q, highlight_c, unified_c = highlighting_from_alignments(final_alignments)

    return scores, highlight_q, highlight_c, unified_c
