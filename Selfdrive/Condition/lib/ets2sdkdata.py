from enum import Enum, auto


class Ets2SdkBoolean(Enum):
    CRUISE_CONTROL = 0
    WIPERS = auto()
    PARK_BRAKE = auto()
    MOTOR_BRAKE = auto()
    ELECTRIC_ENABLED = auto()
    ENGINE_ENABLED = auto()

    BLINKER_LEFT_ACTIVE = auto()
    BLINKER_RIGHT_ACTIVE = auto()
    BLINKER_LEFT_ON = auto()
    BLINKER_RIGHT_ON = auto()

    LIGHTS_PARKING = auto()
    LIGHTS_BEAM_LOW = auto()
    LIGHTS_BEAM_HIGH = auto()
    LIGHTS_AUX_FRONT = auto()
    LIGHTS_AUX_ROOF = auto()
    LIGHTS_BEACON = auto()
    LIGHTS_BRAKE = auto()
    LIGHTS_REVERSE = auto()

    BATTERY_VOLTAGE_WARNING = auto()
    AIR_PRESSURE_WARNING = auto()
    AIR_PRESSURE_EMERGENCY = auto()
    ADBLUE_WARNING = auto()
    OIL_PRESSURE_WARNING = auto()
    WATER_TEMPERATURE_WARNING = auto()

    TRAILER_ATTACHED = -1


class Ets2SdkData:
    def __init__(self):
        self.time = 0
        self.paused = 0

        # tel_revId
        self.ets2_telemetry_plugin_revision = 0
        self.ets2_version_major = 0
        self.ets2_version_minor = 0

        """
         tel_rev1
        """
        self.flags = None

        # Vehicle dynamics
        self.speed = 0.0
        self.accelerationX = 0.0
        self.accelerationY = 0.0
        self.accelerationZ = 0.0

        self.coordinateX = 0.0
        self.coordinateY = 0.0
        self.coordinateZ = 0.0

        self.rotationX = 0.0
        self.rotationY = 0.0
        self.rotationZ = 0.0

        # Drivetrain essentials
        self.gear = 0
        self.gears = 0
        self.gearRanges = 0
        self.gearRangeActive = 0

        self.engineRpm = 0.0
        self.engineRpmMax = 0.0

        self.fuel = 0.0
        self.fuelCapacity = 0.0
        # Not working
        self.fuelRate = 0.0
        self.fuelAvgConsumption = 0.0

        # User input
        self.userSteer = 0.0
        self.userThrottle = 0.0
        self.userBrake = 0.0
        self.userClutch = 0.0

        self.gameSteer = 0.0
        self.gameThrottle = 0.0
        self.gameBrake = 0.0
        self.gameClutch = 0.0

        # Truck & trailer
        self.truckWeight = 0.0
        self.trailerWeight = 0.0

        self.modelOffset = 0
        self.modelLength = 0

        self.trailerOffset = 0
        self.trailerLength = 0

        """
         tel_rev2
        """
        self.timeAbsolute = 0
        self.gearsReverse = 0

        # Trailer ID & display name
        self.trailerMass = 0.0
        self.trailerId = None
        self.trailerName = None

        # Job information
        self.jobIncome = 0
        self.jobDeadline = 0
        self.jobCitySource = None
        self.jobCityDestination = None
        self.jobCompanySource = None
        self.jobCompanyDestination = None

        """
         tel_rev3
        """

        self.retarderBrake = 0
        self.shifterSlot = 0
        self.shifterToggle = 0
        self.fill = 0

        self.aux = None

        self.airPressure = 0.0
        self.brakeTemperature = 0.0

        self.fuelWarning = 0

        self.adblue = 0.0
        self.adblueConsumption = 0.0
        self.oilPressure = 0.0
        self.oilTemperature = 0.0
        self.waterTemperature = 0.0
        self.batteryVoltage = 0.0
        self.lightsDashboard = 0.0
        self.wearEngine = 0.0
        self.wearTransmission = 0.0
        self.wearCabin = 0.0
        self.wearChassis = 0.0
        self.wearWheels = 0.0
        self.wearTrailer = 0.0
        self.truckOdometer = 0.0
        self.cruiseControlSpeed = 0.0

        self.truckMake = None
        self.truckMakeId = None
        self.truckModel = None

        """
         tel_rev4
        """
        self.speedlimit = 0.0

        self.routeDistance = 0.0
        self.routeTime = 0.0

        self.fuelRange = 0.0

        self.gearRatioForward = 0.0
        self.gearRatioReverse = 0.0
        self.gearRatioDifferential = 0.0

        self.gearDashboard = 0

        """
         tel_rev5
        """
        self.onJob = None
        self.jobFinished = None

    def get_boolean(self, b: Ets2SdkBoolean):

        if b == Ets2SdkBoolean.ELECTRIC_ENABLED:
            return self.flags[0] != 0
        elif b == Ets2SdkBoolean.TRAILER_ATTACHED:
            return self.flags[1] != 0
        else:
            return self.aux[b.value] != 0
