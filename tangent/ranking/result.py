__author__ = 'KMDC'

import xml
from tangent.math.symboltree import SymbolTree
from tangent.math.mathsymbol import MathSymbol
from tangent.math.math_extractor import MathExtractor

class Result:
    def __init__(self, query, expression, original_ranking, original_score, mathml=None):
        self.query = query
        self.original_ranking = original_ranking
        self.original_score = original_score
        self.mathml = mathml
        self.new_scores = [0.0]

        if mathml is not None:
            # parse from mathml (additional information extracted)
            self.tree = MathExtractor.convert_and_link_mathml(mathml)
            self.expression = self.tree.tostring()
        else:
            # parse from SLT string (no mathml information available)
            self.tree = SymbolTree.parse_from_slt(expression)
            self.expression = expression

        # print(self.tree.tostring() == expression)

        self.locations = []
        self.matched_elements = []
        self.unified_elements = []
        self.times_rendered = 0

    def set_matched_elements(self, matched_elements):
        self.matched_elements = matched_elements

    def set_unified_elements(self, unified_elements):
        self.unified_elements = unified_elements

    def __modify_xml_ids(self, root):

        if "id" in root.attrib:
            if self.times_rendered == 0:
                root.attrib["id"] += "_0"
            else:
                base_id = root.attrib["id"][:-(len(str(self.times_rendered - 1)) + 1)]
                root.attrib["id"] = base_id + "_" + str(self.times_rendered)

        for child in root:
            self.__modify_xml_ids(child)

    def get_highlighted_mathml(self):

        if self.mathml is not None and self.tree.xml_root is not None:

            if self.times_rendered == 0:
                # compute highlighted mathml only once ...
                self.tree.root.mark_matches("", self.matched_elements, self.unified_elements)

            # modify ids to avoid Math Jax error on expressions rendered multiple times
            self.__modify_xml_ids(self.tree.xml_root)

            marked_mathml = xml.etree.ElementTree.tostring(self.tree.xml_root)
            if isinstance(marked_mathml, bytes):
                marked_mathml = marked_mathml.decode('UTF-8')

            self.highlighted_mathml = marked_mathml

            # count how many times the expression has been rendered...
            self.times_rendered += 1

            return self.highlighted_mathml
        else:
            return None
