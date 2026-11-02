def answer_type(question):
    q = question.lower()
    if re.search(r"\bat what time\b|\bwhat time\b", q):
        return "TIME"
    if re.search(r"\bwhat date\b|\bwhich date\b|\bwhen did\b|\bearliest date\b", q):
        return "DATE"
    if re.search(r"how long|how far|how often|how many (?:seconds|minutes|hours|weeks|months|years|days)|minimum hold|wait time", q):
        return "DURATION"
    if re.search(r"how much|what (?:\w+ ){0,3}(?:price|fee|charge|bill|deposit|credit|refund|rate|amount|cost|rebate)\b|what did .* (?:cost|rise|increase)", q):
        return "MONEY"
    if re.search(r"how many|how old", q):
        return "COUNT"
    return "ENTITY"
