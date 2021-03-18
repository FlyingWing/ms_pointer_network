import re
from typing import Dict, Optional, Iterable, Union, List,Iterator

from overrides import overrides
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.tokenizers import Token
# from allennlp.data.instance import Instance
from allennlp.data.fields import TextField, MetadataField, LabelField
from allennlp.data import DatasetReader, Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

from os import PathLike
PathOrStr = Union[PathLike, str]
DatasetReaderInput = Union[PathOrStr, List[PathOrStr], Dict[str, PathOrStr]]


@DatasetReader.register("ms")
class MSDatasetReader(DatasetReader):
    def __init__(self, namespace='tokens', **args) -> None:
        super().__init__(**args)
        self._source1_token_indexers = {"tokens": SingleIdTokenIndexer(namespace=namespace)}
        self._source2_token_indexers = {"tokens": SingleIdTokenIndexer(namespace=namespace)}
        self._target_token_indexers  = {"tokens": SingleIdTokenIndexer(namespace=namespace)}

    def seg_split(self, query):
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
            
        return tokens


    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # src1, src2, tgt = line.split("\t")
            #src1, src2, tgt = src1.strip(), src2.strip(), tgt.strip()

            tokens = line.split("\t\t")
            if len(tokens) != 4:
                continue
            src1 = ' '.join(self.seg_split(tokens[0]))
            src2 = ' '.join(self.seg_split(tokens[2])) + ' ' + ' '.join(self.seg_split(tokens[3]))
            tgt = ' '.join(self.seg_split(tokens[1]))
            
            if not src1 or not src2 or not tgt:
                continue
            yield self.text_to_instance(src1, src2, tgt)

        
    @overrides
    def text_to_instance(self,
                         source1: str,
                         source2: str,
                         target: Optional[str] = None) -> Instance:
        # 对source1和source2分词、添加END
        source1_tokens = [Token(token) for token in source1.split(" ")]
        source2_tokens = [Token(token) for token in source2.split(" ")]
        source1_tokens.append(Token(END_SYMBOL))
        source2_tokens.append(Token(END_SYMBOL))
        source1_field = TextField(source1_tokens, self._source1_token_indexers)
        source2_field = TextField(source2_tokens, self._source2_token_indexers)

        meta_fields = {
            "source_tokens_1": [x.text for x in source1_tokens[:-1]],
            "source_tokens_2": [x.text for x in source2_tokens[:-1]]
        }
        fields_dict = {
            "source_tokens_1": source1_field,
            "source_tokens_2": source2_field
        }

        if target is not None:
            # 对target分词、添加START、END
            assert all(any(tgt_token == src_token for src_token in source1.split(" ")) or
                       any(tgt_token == src_token for src_token in source2.split(" "))
                       for tgt_token in target.split(" ")), f"target词必须在两个source中出现, {target}, {source1}, {source2}"
            target_tokens = [Token(token) for token in target.split(" ")]
            target_tokens.insert(0, Token(START_SYMBOL))
            target_tokens.append(Token(END_SYMBOL))
            target_field = TextField(target_tokens, self._target_token_indexers)

            fields_dict["target_tokens"] = target_field
            meta_fields["target_tokens"] = [y.text for y in target_tokens[1:-1]]
        
        fields_dict["metadata"] = MetadataField(meta_fields)
        return Instance(fields_dict)
        