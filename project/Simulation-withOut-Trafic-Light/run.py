import traci
traci.start(
    [
        "sumo-gui",
        "-c",
        "../simulation/sumo.sumocfg",
        "--delay",
        "200",
        "--start",
        "true",
        "--xml-validation",
        "never",
        "--log",
        "log",
    ]
)
i = 0
while i < 200:
    traci.simulationStep()
    i += 1
traci.close()    
