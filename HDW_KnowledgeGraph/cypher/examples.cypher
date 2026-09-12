MERGE (sys:System {name: '调压系统'})
MERGE (eq:Equipment {id: 'demo-regulator-station', name: '调压站'})
MERGE (eq)-[:BELONGS_TO]->(sys)
MERGE (fault:Fault {id: 'demo-low-pressure', name: '出口压力偏低'})
MERGE (fault)-[:OCCURS_ON]->(eq);

