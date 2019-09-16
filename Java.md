# Stanford NLP
http://nlp.stanford.edu:8080/parser/index.jsp

Download here: https://nlp.stanford.edu/software/lex-parser.html#Download
```bash
java -mx2g -cp E:\software\stanford-parser-full-2018-10-17\* edu.stanford.nlp.parser.lexparser.LexicalizedParser -nthreads 4 -outputFormat "wordsAndTags,typedDependencies"  -outputFormatOptions "basicDependencies" edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz <file_to_parse>
```