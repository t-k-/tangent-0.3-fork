from subprocess import call

__author__ = 'KMDC'

import io
import xml

from tangent.math.symboltree import SymbolTree
from tangent.math.mathsymbol import MathSymbol
from tangent.math.math_extractor import MathExtractor
from tangent.math.mathml import MathML
from tangent.ranking.result import Result
from tangent.ranking.document_rank_info import DocumentRankInfo
from tangent.ranking.constraint_info import ConstraintInfo

class Query:
    HTML_ResultColumns = 3

    def __init__(self, name, expression, mathml=None, initRetrievalTime='undefined'):
        self.name = name

        self.mathml = mathml
        self.results = {}
        self.documents = {}

        if mathml is not None:
            # parse from mathml (additional information extracted)
            self.tree = MathExtractor.convert_and_link_mathml(mathml)
            self.expression = self.tree.tostring()
        else:
            # parse from SLT string (no mathml information available)
            self.tree = SymbolTree.parse_from_slt(expression)
            self.expression = expression

        self.constraints = Query.create_default_constraints(self.tree)

        self.sorted_results = None
        self.sorted_result_index = None
        self.sorted_abs_ranks = None
        self.sorted_documents = None
        self.sorted_document_index = None
        self.elapsed_time = 0.0

        # RZ: add tuple-based retrieval time and other measures.
        self.initRetrievalTime = initRetrievalTime
        self.postings = None
        self.matchedFormulae = None
        self.matchedDocs = None

        # cache ...
        self.html_queryblock = {}

    @staticmethod
    def create_default_constraints(query_tree):
        # duplicate structure ...
        root = Query.duplicate_structure(query_tree.root, "U")
        # now create constraint nodes ....
        Query.convert_to_constraint_tree(root)

        # create and return symbol tree
        return SymbolTree(root)

    @staticmethod
    def duplicate_structure(current_root, default_tag):
        duplicated_node = MathSymbol(default_tag)

        if current_root.next is not None:
            child = Query.duplicate_structure(current_root.next, default_tag)
            duplicated_node.next = child

        if current_root.above is not None:
            child = Query.duplicate_structure(current_root.above, default_tag)
            duplicated_node.above = child

        if current_root.below is not None:
            child = Query.duplicate_structure(current_root.below, default_tag)
            duplicated_node.below = child

        if current_root.pre_above is not None:
            child = Query.duplicate_structure(current_root.pre_above, default_tag)
            duplicated_node.pre_above = child

        if current_root.pre_below is not None:
            child = Query.duplicate_structure(current_root.pre_below, default_tag)
            duplicated_node.pre_below = child

        if current_root.within is not None:
            child = Query.duplicate_structure(current_root.within, default_tag)
            duplicated_node.within = child

        if current_root.element is not None:
            child = Query.duplicate_structure(current_root.element, default_tag)
            duplicated_node.element = child

        return duplicated_node

    @staticmethod
    def convert_to_constraint_tree(current_root):
        # replace "text" constraint with structure used for restrictions ...
        current_root.tag = ConstraintInfo.create_from_string(current_root.tag)

        if current_root.next is not None:
            Query.convert_to_constraint_tree(current_root.next)

        if current_root.above is not None:
            Query.convert_to_constraint_tree(current_root.above)

        if current_root.below is not None:
            Query.convert_to_constraint_tree(current_root.below)

        if current_root.pre_above is not None:
            Query.convert_to_constraint_tree(current_root.pre_above)

        if current_root.pre_below is not None:
            Query.convert_to_constraint_tree(current_root.pre_below)

        if current_root.within is not None:
            Query.convert_to_constraint_tree(current_root.within)

        if current_root.element is not None:
            Query.convert_to_constraint_tree(current_root.element)

    @staticmethod
    def tree_size(current_root):

        count = 1
        if current_root.next is not None:
            count += Query.tree_size(current_root.next)

        if current_root.above is not None:
            count += Query.tree_size(current_root.above)

        if current_root.below is not None:
            count += Query.tree_size(current_root.below)

        if current_root.pre_above is not None:
            count += Query.tree_size(current_root.pre_above)

        if current_root.pre_below is not None:
            count += Query.tree_size(current_root.pre_below)

        if current_root.within is not None:
            count += Query.tree_size(current_root.within)

        if current_root.element is not None:
            count += Query.tree_size(current_root.element)

        return count


    def set_constraints(self, slt_string):
        # create the tree with the original text labels
        tree_constraints = SymbolTree.parse_from_slt(slt_string)
        # convert the text labels to constraints
        Query.convert_to_constraint_tree(tree_constraints.root)

        if not Query.equal_subtree_structure(self.tree.root, tree_constraints.root):
            print("Warning: Invalid constraint tree specified for " + self.name)
        else:
            self.constraints = tree_constraints

    def add_result(self, doc_id, doc_name, location, expression, score, mathml=None):
        # first, verify if the expression is new...
        if expression not in self.results:
            # new, create....
            ranking = len(self.results) + 1     # assume results are added in original ranking order
            self.results[expression] = Result(self, expression, ranking, score, mathml)

        # add location..
        self.results[expression].locations.append((doc_id, location))

        # add document ...
        if doc_id not in self.documents:
            self.documents[doc_id] = doc_name

    def equal_matched_elements(self, expression1, expression2):
        matched_1 = self.results[expression1].matched_elements
        matched_2 = self.results[expression2].matched_elements

        if len(matched_1) != len(matched_2):
            return False
        else:
            s1 = set(matched_1.keys())
            s2 = set(matched_2.keys())

            return s1 == s2

    def equal_unified_elements(self, expression1, expression2):
        unified_1 = self.results[expression1].unified_elements
        unified_2 = self.results[expression2].unified_elements

        if len(unified_1) != len(unified_2):
            return False
        else:
            s1 = set(unified_1.keys())
            s2 = set(unified_2.keys())

            return s1 == s2

    @staticmethod
    def equal_subtree_structure(root1, root2):
        # a
        if (root1.above is not None) and (root2.above is not None):
            if not Query.equal_subtree_structure(root1.above, root2.above):
                return False
        elif not (root1.above is None and root2.above is None):
            return False
        # b
        if (root1.below is not None) and (root2.below is not None):
            if not Query.equal_subtree_structure(root1.below, root2.below):
                return False
        elif not (root1.below is None and root2.below is None):
            return False
        # c
        if (root1.pre_above is not None) and (root2.pre_above is not None):
            if not Query.equal_subtree_structure(root1.pre_above, root2.pre_above):
                return False
        elif not (root1.pre_above is None and root2.pre_above is None):
            return False
        # d
        if (root1.pre_below is not None) and (root2.pre_below is not None):
            if not Query.equal_subtree_structure(root1.pre_below, root2.pre_below):
                return False
        elif not (root1.pre_below is None and root2.pre_below is None):
            return False
        # n
        if (root1.next is not None) and (root2.next is not None):
            if not Query.equal_subtree_structure(root1.next, root2.next):
                return False
        elif not (root1.next is None and root2.next is None):
            return False
        # w
        if (root1.within is not None) and (root2.within is not None):
            if not Query.equal_subtree_structure(root1.within, root2.within):
                return False
        elif not (root1.within is None and root2.within is None):
            return False
        # e
        if (root1.element is not None) and (root2.element is not None):
            if not Query.equal_subtree_structure(root1.element, root2.element):
                return False
        elif not (root1.element is None and root2.element is None):
            return False

        # leaf or same sub-structure found ...
        return True


    def equal_structure(self, expression1, expression2):
        tree1 = self.results[expression1].tree.root
        tree2 = self.results[expression2].tree.root

        return Query.equal_subtree_structure(tree1, tree2)

    def sort_documents(self):
        if len(self.results) > 0:
            current_documents = {}

            # sum scores for all existing formulas over all documents
            for expression in self.results:
                result = self.results[expression]

                for doc_id, location in result.locations:
                    # add document if first time seen
                    if not doc_id in current_documents:
                        current_documents[doc_id] = DocumentRankInfo(doc_id)

                    # add score of current result to current document
                    current_documents[doc_id].add_formula_scores(expression, location, result.new_scores)

            all_docs = [((current_documents[doc_id].top_formula_score, current_documents[doc_id].total_score),
                         current_documents[doc_id]) for doc_id in current_documents]

            all_docs = sorted(all_docs, key=lambda x: x[0], reverse=True)

            self.sorted_documents = [doc for scores, doc in all_docs]

            self.sorted_document_index = {}
            for idx, doc in enumerate(self.sorted_documents):
                self.sorted_document_index[doc.doc_id] = idx


    def sort_results(self):
        if len(self.results) > 0:
            # different number of score might be in use...
            n_scores = len(self.results[next(iter(self.results))].new_scores)
            score_function = lambda x: [x[i] for i in range(n_scores)]

            # now, sort them ...
            result_list = [self.results[expression].new_scores + [expression] for expression in self.results]
            sorted_list = [x[-1] for x in sorted(result_list, key=score_function, reverse=True)]

            # first group by unique scores...
            last_group_scores = None
            sorted_groups = []
            for expression in sorted_list:
                if last_group_scores is None or last_group_scores != self.results[expression].new_scores:
                    # create new group
                    sorted_groups.append([])
                    last_group_scores = self.results[expression].new_scores

                sorted_groups[-1].append(expression)

            # now, create sub groups based on same structure matched....
            for group_idx in range(len(sorted_groups)):
                group_list = sorted_groups[group_idx]

                # find subgroups with the same matching substructure ...
                sub_group_list = []

                # ... for the remaining elements ...
                for expression in group_list:
                    # ... compare againts every group ...
                    found = False
                    for group in sub_group_list:
                        group_expression = group[0]

                        # ... compare match structure and match location...
                        if (self.equal_structure(expression, group_expression) and
                            self.equal_matched_elements(expression, group_expression) and
                            self.equal_unified_elements(expression, group_expression)):

                            group.append(expression)
                            found = True
                            break

                    if not found:
                        sub_group_list.append([expression])

                # finally, replace...
                sorted_groups[group_idx] = sub_group_list

            # self.sorted_results = [x[-1] for x in sorted(result_list, key=score_function, reverse=True)]
            self.sorted_results = sorted_groups

            # keep an inverted index from expressions to their sub groups indices after sorting
            self.sorted_result_index = {}
            self.sorted_abs_ranks = {}
            sub_group_idx = 0
            previous_count = 0

            for group in self.sorted_results:
                current_count = 0
                for subgroup in group:
                    sub_group_idx += 1
                    current_count += len(subgroup)

                    # for expression in self.sorted_results:
                    for expression in subgroup:
                        self.sorted_result_index[expression] = sub_group_idx
                        self.sorted_abs_ranks[expression] = previous_count

                previous_count += current_count



    def get_query_stats(self):
        total_matches = 0
        total_formulae = 0

        for group in self.sorted_results:
            total_matches += len(group)
            for subgroup in group:
                total_formulae += len(subgroup)

                for expression in subgroup:
                    result = self.results[expression]

        total_documents = len(self.documents)

        return total_matches, total_formulae, total_documents


    def output_query(self, out_file):
        out_file.write("Q\t" + self.name + "\n")
        out_file.write("E\t" + self.expression + "\n")

    def output_sorted_results(self, out_file):
        if self.sorted_results is None:
            print("Results must be sorted first: output_sorted_results")
            return

        for group in self.sorted_results:
            for subgroup in group:

                # for expression in self.sorted_results:
                for expression in subgroup:
                    result = self.results[expression]
                    for doc_id, location in result.locations:
                        out_file.write("R\t" + str(doc_id) + "\t" + str(location) + "\t" + result.expression + "\t" +
                                       str(result.new_scores[0]) + "\n")

    def output_stats(self, out_file, separator, test_condition):
        if self.sorted_results is None:
            print("Results must be sorted first: output_stats")
            return

        q_size = Query.tree_size(self.tree.root)

        structure_idx = 0
        for g_idx, group in enumerate(self.sorted_results):
            for subgroup in group:
                structure_idx += 1

                for expression in subgroup:
                    result = self.results[expression]

                    c_size = Query.tree_size(result.tree.root)

                    values = [self.name, test_condition, str(result.original_ranking), str(result.original_score),
                              str(g_idx + 1), str(structure_idx), str(result.new_scores[0]), str(result.new_scores[1]),
                              str(result.new_scores[2]), str(q_size), str(c_size), result.expression]
                    line = separator.join(values)
                    out_file.write(line + "\n")

    @staticmethod
    def stats_header(separator):
        header = separator.join(["query", "condition", "o_rank", "o_score", "n_rank", "n_str",
                                 "n_score_1", "n_score_2", "n_score_3", "q_size", "c_size", "slt"])
        return header + "\n"


    def save_png(self, output_name, tree, highlight_nodes=None, unified_nodes=None, generic=False):
        # first, save to temporal dot file
        tree.save_as_dot("temporal_rerank_graph.gv", highlight_nodes, unified_nodes, generic)

        # now, execute dot....
        try:
            code = call(["dot", "-Tpng", "temporal_rerank_graph.gv", "-o", output_name])
        except:
            print("Must install dot in order to use HTML output")
            return False

        return code == 0

    def save_svg(self, output_name, tree, highlight_nodes=None, unified_nodes=None, generic=False):
        # first, save to temporal dot file
        tree.save_as_dot("temporal_rerank_graph.gv", highlight_nodes, unified_nodes, generic)

        # now, execute dot....
        try:
            code = call(["dot", "-Tsvg", "temporal_rerank_graph.gv", "-o", output_name])
        except:
            print("Must install dot in order to use HTML output")
            return False

        return code == 0

    def __recursive_find_elements(self, current_root, tag):
        if current_root.tag == tag:
            result = [current_root]
        else:
            result = []

        for element in current_root:
            child_res = self.__recursive_find_elements(element, tag)
            result += child_res

        return result

    def get_html_common_header(self):
        return """
        <!DOCTYPE html>
        <html>
            <head>
                <title>Results for: """ + self.name + """</title>
                <style>
                 .results_list td  { border: 0px solid black; padding: 5px; }
                 .results_list th  { border: 0px solid black; padding: 5px; }

                 .math_formula {
                     background-color: #fff;
                     padding: 10px;
                     border: 1px solid #ddd;
                     font-size: 200%;
                     font-family: Helvetica;
                 }

                 #body {
                    margin: 0;
                    font-family: "Helvetica Neue";
                    font-size: 1em;
                    color: #222;
                    padding: 30px 60px;
                }

                #statsline {
                    font-size: 1.25em;
                }


                #logo {
                    width: 140px;
                    font-family: "Helvetica Neue";
                    font-weight: 250;
                    font-size: 2em;
                    float: left;
                }

                .score {
                    color: #999;
                }

                #queryblock {
                }

                #header {
                    background-color: #eee;
                    padding: 12px 30px;
                    overflow: auto;
                    border-bottom: 1px solid #ccc;
                }

                #searchbutton {
                    padding: 5px;
                    background-color: #efe;
                    border: 2px solid black;
                    height: 2.5em;
                    width: 6.5em;
                    font-size: 125%;
                }

                </style>
                <meta charset=\"UTF-8\">
                <script type=\"text/javascript\"
                   src=\"http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML\">
                </script>
                <script type=\"text/javascript\" >
                    function hide_class(hide, class_name){
                        var nodes = document.getElementsByClassName(class_name);
                        var i;
                        for (i = 0; i < nodes.length; i++){
                            if (hide){
                                nodes[i].style.display = "none";
                            } else {
                                nodes[i].style.display = "";
                            }
                        }

                        if (hide){
                            document.getElementById('id_' + class_name + '_show').style.display = '';
                            document.getElementById('id_' + class_name + '_hide').style.display = 'none';
                        } else {
                            document.getElementById('id_' + class_name + '_show').style.display = 'none';
                            document.getElementById('id_' + class_name + '_hide').style.display = '';
                        }
                    }
                </script>
            </head>
            <body>
                <font face=helvetica>
        """

    def get_html_common_footer(self):
        return """
                </font>
            </body>
        </html>
        """

    def get_html_queryblock(self, prefix):
        if prefix not in self.html_queryblock:
            # create the SVG for the query ...

            image_base_name = "images/" + self.name + "_"

            if self.save_svg(prefix + "/" + image_base_name + "query.svg", self.tree):
                query_image = """
                <object type=\"image/svg+xml\" data=\"""" + image_base_name + """query.svg\">
                    Not Supported
                </object>"""
            else:
                query_image = """
                    <p><Query SLT could not be rendered</p>
                """

            # Prepare MathML
            if self.mathml is not None:
                if isinstance(self.mathml, bytes):
                    self.mathml = self.mathml.decode('UTF-8')

                # remove qvar elements....
                elem_content = io.StringIO(self.mathml)  # treat the string as if a file
                root = xml.etree.ElementTree.parse(elem_content).getroot()
                all_vars = self.__recursive_find_elements(root, MathML.mqvar)
                all_vars += self.__recursive_find_elements(root, MathML.mqvar2)

                if len(all_vars) > 0:
                    for query_var in all_vars:
                        query_var.tag = MathML.mi
                        query_var.text = query_var.attrib["name"]

                    query_mathml = xml.etree.ElementTree.tostring(root)
                    if isinstance(query_mathml, bytes):
                        query_mathml = query_mathml.decode('UTF-8')
                else:
                    query_mathml = self.mathml
            else:
                query_mathml = ""

            self.html_queryblock[prefix] = """
            <!-- Query -->
            <div id="queryblock" align="left">
                <table>
                    <tr><td>
                        <div class="tree_svg" style="display: none;">""" + query_image + """</div>
                    </td></tr>
                    <tr><td>
                        <div class=\"math_formula\">""" + query_mathml + """</div>
                    </td></tr>
                </table>
            </div>
            """

        return self.html_queryblock[prefix]

    def get_html_logo(self, prefix, include_show_buttons):
        result = """
        <div id="header">
            <table><tr>
                <td>
                    <!-- Logo and buttons -->
                    <div>
                        <div id="logo">tangent<br>

                            <table align="left" ><tr>
                                <td>
        """

        if include_show_buttons:
            result += """
            <input type="button" id="id_tree_svg_show" value="Graphs" onclick="hide_class(false, 'tree_svg');">
            <input type="button" id="id_tree_svg_hide" value="Graphs" style="background:yellow; display: none;"
                onclick="hide_class(true, 'tree_svg');">
            """
        else:
            result += "<br />"

        result += """
                                </td>
                            </tr></table>
                        </div>
                    </div>
                <td>""" + self.get_html_queryblock(prefix) + """</td>
                <td width="99%" align="right">
                    <!-- Search Button -->
                    <button id="searchbutton" type="button">Search</button>
                </td>
            </tr></table>
        </div>
        """

        return result

    def get_html_stats(self):
        # compute statistics ....
        total_matches, total_formulae, total_locations = self.get_query_stats()

        result = """
        <!-- STATISTICS -->
        <div id="statsline">
            Returned """ + str(total_matches) + """ matches
            (""" + str(total_formulae) + """ formulae, """ + str(total_locations) + """ docs)
            <br>&nbsp;&nbsp;&nbsp;&nbsp;Lookup """ + "{0:.3f}".format(self.initRetrievalTime) + " ms, Re-ranking " +  "{0:.3f}".format(self.elapsed_time) + """ ms<br>&nbsp;&nbsp;&nbsp;&nbsp;Found """ + str(self.postings) + """ tuple postings, """ + str(self.matchedFormulae) + """ formulae, """ + str(self.matchedDocs) + """ documents
            <br>
            <table cellpadding="5">
                <tr>
                    <td>
                        <A href=\"""" + self.name + """_main.html\" style="text-decoration:none">
                            [ formulas ]
                        </a>
                    </td>
                    <td>
                        <A href=\"""" + self.name + """_docs.html\" style="text-decoration:none">
                            [ documents ]
                        </a>
                    </td>
                    <td>
                        <A href=\"""" + self.name + """_formulas.html\" style="text-decoration:none">
                            [ documents-by-formula ]
                        </a>
                    </td>
                </tr>
            </table>
            <br>
        </div>
        """

        return result

    def save_html_groups(self, prefix):
        if self.sorted_results is None:
            print("Results must be sorted first: save_html_groups")
            return False

        base_name = prefix + "/" + self.name
        out_filename = base_name + "_main.html"
        image_base_name = "images/" + self.name + "_"

        header = self.get_html_common_header()
        header += self.get_html_logo(prefix, True)

        content = "<div id=\"body\">"
        content += self.get_html_stats()

        content += "<table class=\"results_list\" align=\"left\" cellpadding=\"0\" cellspacing=\"0\" border=\"0\" >\n"

        g_idx = 0
        exp_idx = 0
        for group in self.sorted_results:
            for subgroup in group:
                g_idx += 1

                # output header row
                content += "<tr>"

                # scores...
                # ... add an anchor to current group
                scores_str = "<br />".join(["{0:.4f}".format(x) for x in self.results[subgroup[0]].new_scores])
                content += """<td rowspan=\"2\" style=\"vertical-align: text-top;\" >
                                <a name=\"group_""" + str(g_idx) + """\"></a>
                                <div class="score">""" + scores_str + """</div>
                               </td>
                            """

                # SVG tree
                content += "<td>"
                result_name = image_base_name + str(g_idx) + ".svg"
                first_result = self.results[subgroup[0]]

                if self.save_svg(prefix + "/" + result_name, first_result.tree, first_result.matched_elements,
                                 first_result.unified_elements, True):
                    content += "<object class=\"tree_svg\" style=\"display: none;\" type=\"image/svg+xml\" " \
                               "data=\"" + result_name + "\">Not Supported</object>"

                content += "</td>"
                content += "</tr>\n"

                # now expressions in single cell (on their own tables)
                content += "<tr><td>"


                content += "<table>"
                for sg_idx, expression in enumerate(subgroup):
                    exp_idx += 1

                    result = self.results[expression]

                    if sg_idx % Query.HTML_ResultColumns == 0:
                        content += "<tr>"

                    #content += "<td>" + str(sg_idx + 1) + "</td>"
                    content += "<td></td>"
                    content += "<td>"
                    marked_mathml = result.get_highlighted_mathml()
                    if marked_mathml is not None:
                        content += "<a href=\"" + self.name + "_formulas.html#formula_" + str(exp_idx) + "\"  >"
                        content += "    <div class=\"math_formula\">" + marked_mathml + "</div>"
                        content += "</a>"
                    else:
                        content += "<br />"
                    content += "</td>\n"

                    if (sg_idx + 1) % Query.HTML_ResultColumns == 0:
                        content += "</tr>\n"

                # create the empty cell for the empty spaces in the grid of results...
                reminder = len(subgroup) % Query.HTML_ResultColumns
                if reminder > 0:
                    content += "<td colspan=\"" + str((Query.HTML_ResultColumns - reminder) * 2) + "\"><br /></td>"
                    content += "</tr>\n"

                content += "</table>"
                content += "</td></tr>"


        content += "</table>\n"

        content += "</div>\n"

        footer = self.get_html_common_footer()

        out_file = open(out_filename, "wb")
        final_content = bytes(header + content + footer, "UTF-8")
        out_file.write(final_content)
        out_file.close()

        return True


    def save_html_docs(self, prefix):
        if self.sorted_documents is None:
            print("Documents must be sorted first: save_html_docs")
            return False

        base_name = prefix + "/" + self.name
        out_filename = base_name + "_docs.html"
        # image_base_name = "images/" + self.name + "_"

        header = self.get_html_common_header()
        header += self.get_html_logo(prefix, False)

        content = "<div id=\"body\">"
        content += self.get_html_stats()

        content += "<table class=\"results_list\" align=\"left\" cellpadding=\"0\" cellspacing=\"0\" border=\"0\" >\n"
        for idx, document in enumerate(self.sorted_documents):

            doc_link = self.documents[document.doc_id]
            doc_params = ""
            for loc, expr in document.expressions:
                doc_params += "&exp=" + str(loc)
                #doc_params += "&int=" + str(self.results[expr].new_scores[0])
                doc_params += "&int=" + str(1.0 - (self.sorted_abs_ranks[expr] / float(len(self.results))))

            content += "<tr>"
            content += "<td rowspan=\"2\" style=\"vertical-align: text-top;\">Doc " + str(idx + 1) + "</td>"

            scores_str = document.get_score_string("<br />")
            content += """<td rowspan=\"2\" style=\"vertical-align: text-top;\" >
                            <div class="score">""" + scores_str + """</div>
                           </td>
                        """
            content += "<td><a href=\"../highlighter.html?doc=" + doc_link + doc_params + "\">" + doc_link + "</a></td>"
            content += "</tr>"

            content += "<tr>"
            content += "<td>"
            content += "<table>"

            sorted_locs = sorted([(self.sorted_result_index[expr], expr) for loc, expr in document.expressions])

            for sg_idx, exp_info in enumerate(sorted_locs):
                group_idx, expression = exp_info
                result = self.results[expression]

                if sg_idx % Query.HTML_ResultColumns == 0:
                    content += "<tr>"

                content += "<td></td>"
                content += "<td>"
                marked_mathml = result.get_highlighted_mathml()
                if marked_mathml is not None:

                    content += "<a href=\"" + self.name + "_main.html#group_" + str(group_idx) + "\"  >"
                    content += "    <div class=\"math_formula\">" + marked_mathml + "</div>"
                    content += "</a>"
                else:
                    content += "<br />"
                content += "</td>\n"

                if (sg_idx + 1) % Query.HTML_ResultColumns == 0:
                    content += "</tr>\n"

            # create the empty cell for the empty spaces in the grid of results...
            reminder = len(document.expressions) % Query.HTML_ResultColumns
            if reminder > 0:
                content += "<td colspan=\"" + str((Query.HTML_ResultColumns - reminder) * 2) + "\"><br /></td>"
                content += "</tr>\n"

            content += "</table>"
            content += "</td>"
            content += "</tr>"

        content += "</table>\n"

        content += "</div>\n"

        footer = self.get_html_common_footer()

        out_file = open(out_filename, "wb")
        final_content = bytes(header + content + footer, "UTF-8")
        out_file.write(final_content)
        out_file.close()

        return True

    def save_html_formulas(self, prefix):
        if self.sorted_results is None:
            print("Results must be sorted first: save_html_formulas")
            return False

        if self.sorted_documents is None:
            print("Documents must be sorted first: self.sorted_documents is None")
            return False

        base_name = prefix + "/" + self.name
        out_filename = base_name + "_formulas.html"
        # image_base_name = "images/" + self.name + "_"

        header = self.get_html_common_header()
        header += self.get_html_logo(prefix, False)

        content = "<div id=\"body\">"
        content += self.get_html_stats()

        content += "<table class=\"results_list\" align=\"left\" cellpadding=\"0\" cellspacing=\"0\" border=\"0\" >\n"
        exp_idx = 0
        for group in self.sorted_results:
            for subgroup in group:
                for sg_idx, expression in enumerate(subgroup):
                    exp_idx += 1

                    group_idx = self.sorted_result_index[expression]

                    result = self.results[expression]

                    # MathML
                    content += "<tr>\n<td>\n"
                    content += "<a name=\"formula_" + str(exp_idx) + "\"></a>"
                    content += "<table><tr><td>\n"
                    marked_mathml = result.get_highlighted_mathml()
                    if marked_mathml is not None:
                        content += "<a href=\"" + self.name + "_main.html#group_" + str(group_idx) + "\"  >"
                        content += "<div class=\"math_formula\">" + marked_mathml + "</div>"
                        content += "</a>"
                    else:
                        content += "<br />"
                    content += "</td></tr></table>\n"
                    content += "</td>\n</tr>\n"

                    # Document lists....
                    content += "<tr>\n<td>\n"

                    content += "<table>"
                    sorted_idxs = sorted([self.sorted_document_index[doc_id] for doc_id, loc in result.locations])
                    for sorted_idx in sorted_idxs:
                        document = self.sorted_documents[sorted_idx]
                        doc_id = document.doc_id
                        doc_link = self.documents[doc_id]

                        doc_params = ""
                        for loc, expr in document.expressions:
                            doc_params += "&exp=" + str(loc)
                            #doc_params += "&int=" + str(self.results[expr].new_scores[0])
                            doc_params += "&int=" + str(1.0 - (self.sorted_abs_ranks[expr] / float(len(self.results))))

                        content += "<tr>"
                        content += "<td>Doc " + str(sorted_idx + 1) + "</td>"
                        scores_str = document.get_score_string(", ")
                        content += """
                                    <td style=\"vertical-align: text-top;\" >
                                        <div class="score">""" + scores_str + """</div>
                                    </td>
                                    """
                        link = "<a href=\"../highlighter.html?doc=" + doc_link + doc_params + "\">" + doc_link + "</a>"
                        content += "<td>" + link + "</td>"
                        content += "</tr>"
                    content += "</table>"

                    content += "</td>\n</tr>\n"

        content += "</table>\n"

        content += "</div>\n"

        footer = self.get_html_common_footer()

        out_file = open(out_filename, "wb")
        final_content = bytes(header + content + footer, "UTF-8")
        out_file.write(final_content)
        out_file.close()

        return True

    def save_html(self, prefix):

        # save the main page with groups ....
        if not self.save_html_groups(prefix):
            return False

        # save the docs page ...
        if not self.save_html_docs(prefix):
            return False

        # save the formulas page ...
        if not self.save_html_formulas(prefix):
            return False

        # all pages saved successfully
        return True




