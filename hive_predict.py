import argparse
import math
import re
import sys
from typing import Any, Union, Dict, Iterable, List, Optional, Tuple

from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor

from model import MSPointerNetwork
from dataset_reader import MSDatasetReader
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer, PretrainedBertIndexer

from allennlp.data import vocabulary


from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor


def seg_split(query):
        #pattern = re.compile(r'/|【|】|\*|-|（|）|\(|\)|\[|\]|&|「|」|★|！|▲|x|%|\||｜|#')
        pattern = re.compile(r'[^\.A-Za-z0-9\u4e00-\u9fa5]')
        query = re.sub(pattern,' ',query)
        strinfo = re.compile('[\u4e00-\u9fa5]{1,}')
        # s1 = strinfo.sub(" ", '17哈弗H6豪华')
        s1 = strinfo.split(query)
        s2 = strinfo.findall(query)
        #print(s1)
        #print(s2)

        tokens=[]
        minlen = min(len(s1), len(s2))
        for e1, e2 in zip(s1[:minlen], s2[:minlen]):
            if e1=='':
                tokens += [w for w in e2]
            else:
                if e1 != ' ':
                    tokens  += e1.strip().split()
                tokens += [w for w in e2]

        if len(s1) > len(s2) and s1[-1] != '':
            tokens+=s1[-1].strip().split()
        elif len(s1) < len(s2):
            tokens += [w for w in s2[-1]]

        return ' '.join(tokens)

@Predictor.register('ms_pointer')
class MSPointerPredictor(Predictor):

    def predict(self, source1: str, source2: str) -> JsonDict:
        return self.predict_json({"source1": source1, "source2": source2})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        source1 = json_dict["source1"]
        source2 = json_dict["source2"]
        return self._dataset_reader.text_to_instance(source1, source2)


if __name__ == "__main__":

    archive = load_archive('./outputs/ms_pointer2')
    predictor = Predictor.from_archive(archive=archive, predictor_name="ms_pointer")

    for line in sys.stdin:
        tokens = line.strip("\r\n").split('\t')
        if len(tokens) < 4:
            continue 

        item_id   = tokens[0]
        raw_title = tokens[1]
        brand     = tokens[2]
        entity    = tokens[3]

        if len(raw_title) <= 10:
            print(f'{item_id}\t{raw_title}')
            continue

        with open(file_path, 'r', encoding="utf-8") as file:
            file_data = file.read()
            titleMap = yaml.load(file_data)

        if raw_title in titleMap:
            print(f'{item_id}\t{titleMap[raw_title]}')
            continue

        source1 = seg_split(raw_title)
        source2 = seg_split(brand) + ' ' + seg_split(entity)
        output_dict = predictor.predict(source1, source2)

        oov = [token for token in (source2+" "+source1).split(' ') if predictor._model.vocab.get_token_index(token) == 1]
        oov = sorted(set(oov),key=oov.index)

        res = [oov[0] if token == '@@UNKNOWN@@' else token for token in output_dict["predicted_tokens"][0]]
        sentence = "".join(res)
        predictor._model.vocab.get_token_index("asdf")

        print(f'{item_id}\t{sentence}')
      