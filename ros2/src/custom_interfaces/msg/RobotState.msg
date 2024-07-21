uint8 DISCONNECTED=0
uint8 READY=1
uint8 IDLE=2
uint8 BUSY=3
uint8 RECONNECTING=4
uint8 ERROR=5

# Message definition
uint8 state
uint32 error_id 
string error_code
string error_message
bool robot_ready 
bool tp_enabled 
bool remote_running 
uint32 speed 
string inst_in_progress
