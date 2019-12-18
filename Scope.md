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

Grouping and Aggregation
```sql
rs1 = SELECT 
COUNT() AS NumSessions, 
Region, 
SUM(Duration) AS TotalDuration, 
AVG(Duration) AS AvgDwellTtime, 
MAX(Duration) AS MaxDuration, 
MIN(Duration) AS MinDuration
    FROM searchlog
    GROUP BY Region;
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

# Processor
```csharp
public class ExampleProcessor : Processor
{
    public override Schema Produces(string[] requestedColumns, string[] args, Schema input)
    {
        return new Schema(@"QueryA:string,QueryB:string");
    }

    public override IEnumerable<Row> Process(RowSet input, Row outputRow, string[] args)
    {
        foreach (var row in input.Rows)
        {
            string QueryA = row[0].String;
            string ClicksA = row[1].String;
            string QueryB = row[2].String;
            string ClicksB = row[3].String;

            if (QueryA == QueryB) continue;
            if (!IsSimilar(ClicksA, ClicksB)) continue;

            outputRow[0].Set(QueryA);
            outputRow[1].Set(QueryB);

            yield return outputRow;
        }
    }
}

public class ExampleProcessor2 : Processor
{
    public override Schema Produces(string[] columns, string[] args, Schema input)
    {
        Schema output = input.Clone();
        output.Add(new ColumnInfo("NewColumn1", ColumnDataType.String));
        output.Add(new ColumnInfo("NewColumn2", ColumnDataType.String));
        return output;
    }

    public override IEnumerable<Row> Process(RowSet input, Row outputRow, string[] args)
    {
        foreach (var row in input.Rows)
        {
            row.CopyTo(outputRow);
            string questions = row["Questions"].String;
            string answers = row["Answers"].String;
            outputRow["NewColumn1"].Set(questions);
            outputRow["NewColumn2"].Set(answers);
            yield return outputRow;
        }
    }
}
```

# Reducer
```csharp
QALogs = SELECT EventInfo_Time, UserId, Question, Answer FROM InputLogs;
ReorderQALogs =
    REDUCE QALogs
    ON UserId
    PRODUCE *
    USING ReorderReducer
    PRESORT EventInfo_Time ASC;

#CS
public class ReorderReducer : Reducer
{
    public override Schema Produces(string[] requestedColumns, string[] args, Schema input)
    {
        return new Schema("PrevQuestion:string,PrevAnswer:string,CurrentQuery:string");
    }

    public override IEnumerable<Row> Reduce(RowSet input, Row outputRow, string[] args)
    {
        string prevQuestion = string.Empty;
        string prevAnswer = string.Empty;
        foreach (Row inputRow in input.Rows)
        {
            var question = inputRow[2].ToString();
            var answer = inputRow[3].ToString();
            outputRow[0].Set(prevQuestion);
            outputRow[1].Set(prevAnswer);
            outputRow[2].Set(question);
            prevQuestion = question;
            prevAnswer = answer;
            yield return outputRow;
        }
    }
}
#ENDCS
```

# Random select rows
```sql
// add random id
SELECT *, Guid.NewGuid().ToString() AS RandomId;
// keep only a certain amount
SELECT *, ROW_NUMBER() OVER(PARTITION BY Message ORDER BY RandomId) AS RowNumber HAVING RowNumber <= @MaxCount; 
```

# Get row number
```sql
SELECT *, ROW_NUMBER() OVER() AS RowNumber;
```