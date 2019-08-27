# Basics

Extraction
```sql
rs0 = EXTRACT 
        FirstName : string,
        LastName : string,
        Age : int
      FROM
        "/test_input.tsv"
      USING DefaultTextExtractor("-silent");
```

Tesing for membership with the IN operator
```sql
rs = SELECT FirstName, LastName, JobTitle
     FROM People
     WHERE JobTitle IN ("Design Engineer", "Tool Designer", "Marketing Assistant");
```

Distinct rows
```sql
rs1 = SELECT DISTINCT Region FROM searchLog;
```

# Regular Expressions
Finding Simple Patterns
```sql
// Find all the sessions where the query contained the word pizza (but not pizzeria, for example)

rs1 = 
    SELECT 
          Start,
          Region,
          Duration
    FROM searchlog
    WHERE REGEX(@"\bpizza.*\b").IsMatch(Query);
 
OUTPUT rs1 TO "/my/Outputs/output.txt";
```

Extracting a REGEX Match
```sql
rs1 = 
    SELECT
        Name,
        REGEX(@"Cosmos[0-9]*").Match(Name).Value AS CosmosCluster
    FROM data;
```

# Cross Apply
```sql
rs1 = 
    SELECT 
        Region, 
        Urls
    FROM searchlog;

rs2 = 
    SELECT 
        Region, 
        SplitUrls AS Url
    FROM  rs1
    CROSS APPLY Urls.Split(';') AS SplitUrls;
```

# Stream Set
The following
```sql
data = EXTRACT a:string, b:string
       FROM "/my/SampleData/StreamSets/log1.txt" ,
            "/my/SampleData/StreamSets/log2.txt" ,
            "/my/SampleData/StreamSets/log3.txt" 
       USING DefaultTextExtractor();
```
can be written as:
```sql
data = 
    EXTRACT 
        a:int, 
        b:string
    FROM STREAMSET "/my/SampleData/StreamSets/"
         PATTERN "log%n.txt"
         RANGE __serialnum=["1", "3"];
```

More examples:

__serialnum
```sql
SSTREAM 
    SPARSE STREAMSET @"/shares/bingads.platform.prod.indexgen/Data/publish/SDP-Prod-Ch1/AdIndex_Corpus/"
    PATTERN @"FullAdCorpus.ss.%Year-%Month-%Day_%Hour_%Minute_%Second.%n.ss"
    RANGE __datetime=[@StartDate1,@StartDate2]("00:00:01"), __serialnum=["0","1000"]
```
```sql
// Modern Syntax 
Rs1 = EXTRACT a:int, b:int
      FROM STREAMSET @"/my/Tests/"
           PATTERN @"out_%n.txt"
           RANGE __serialnum=["0", "9"]
      USING DefaultTextExtractor();

// Legacy Syntax
Rs1 = EXTRACT a:int, b:int
      FROM @"/my/Tests/out_%n.txt?serialnum=0...9"
      USING DefaultTextExtractor();
```

__date
```sql
Rs1 = 
    EXTRACT col1:string, col2:int 
    FROM STREAMSET @"D:\ScopeDemo\Input\"
PATTERN "%Y\%m\log_%d.txt"
RANGE __date = ["2013-10-01", "2013-10-03"] 
    USING DefaultTextExtractor();

// Legacy Syntax
Rs1 = 
    EXTRACT col1:string, col2:int 
    FROM @"D:\ScopeDemo\Input\%Y\%m\log_%d.txt?date=2013-10-01...2013-10-03" 
    USING DefaultTextExtractor();
```

__datetime
```sql
// Modern Syntax
Rs1 = 
    EXTRACT col1:string, col2:int 
    FROM STREAMSET @"/my/Tests/"
       PATTERN @"%Year/%Month/%Day_%Hour_%Minute_%Second"
       RANGE __datetime=["2014-03-07T00:00:00", "2014-03-07T20:00:00"];
    USING DefaultTextExtractor();
```

__hour
```sql
// Modern Syntax 
Rs1 = EXTRACT a:int, b:int
      FROM STREAMSET @"/my/Tests/"
           PATTERN @"out_%h.txt"
           RANGE __hour=["0", "9"]
      USING DefaultTextExtractor();

// Legacy Syntax
Rs1 = EXTRACT a:int, b:int
      FROM @"/my/Tests/out_%h.txt?hour=0...9"
      USING DefaultTextExtractor();
```