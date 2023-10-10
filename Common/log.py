from loguru import logger

user_info_level = logger.level("UserInfo", no=10, color="<green>")
user_data_level = logger.level("UserData", no=20, color="<white>")
user_warning_level = logger.level("UserWarning", no=30, color="<yellow>")

planning_info_level = logger.level("PlanningInfo", no=10, color="<green>")
planning_data_level = logger.level("PlanningData", no=20, color="<white>")
planning_warning_level = logger.level("PlanningWarning", no=30, color="<yellow>")

perception_info_level = logger.level("PerceptionInfo", no=10, color="<green>")
perception_data_level = logger.level("PerceptionData", no=20, color="<white>")
perception_warning_level = logger.level("PerceptionWarning", no=30, color="<yellow>")
perception_error_level = logger.level("PerceptionError", no=30, color="<red>")

control_info_level = logger.level("ControlInfo", no=10, color="<green>")
control_data_level = logger.level("ControlData", no=20, color="<white>")
control_warning_level = logger.level("ControlWarning", no=30, color="<yellow>")

condition_info_level = logger.level("ConditionInfo", no=10, color="<green>")
condition_data_level = logger.level("ConditionData", no=20, color="<white>")
condition_warning_level = logger.level("ConditionWarning", no=30, color="<yellow>")
condition_error_level = logger.level("ConditionError", no=30, color="<red>")
