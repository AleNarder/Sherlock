# State definition
uint8 NOT_INIT=0
uint8 INIT=1
uint8 IDLE=2
uint8 RUNNING=3
uint8 PAUSED=4
uint8 STOPPED=5
uint8 HARD_STOPPED=6
uint8 ERROR=99 # o gaucho

# State description
string NOT_INIT_DESC="NOT INITIALIZED"
string INIT_DESC="INITIALIZED"
string IDLE_DESC="IDLE"
string RUNNING_DESC="RUNNING"
string PAUSED_DESC="PAUSED"
string STOPPED_DESC="STOPPED"
string HARD_STOPPED_DESC="HARD STOPPED"
string ERROR_DESC="ERROR"



# Message definition
uint8 state
string state_description