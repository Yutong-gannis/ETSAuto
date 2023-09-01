import mmap
import struct

from lib.ets2sdkdata import Ets2SdkData


class SharedMemory:
    def __init__(self):
        self.map_name = "Local\\SimTelemetryETS2"
        self.map_size = 1024
        self.mmap = None

    def connect(self):
        self.mmap = mmap.mmap(0, self.map_size, self.map_name, mmap.ACCESS_READ)

    def update(self):
        self.connect()
        updated_data = Ets2SdkData()

        updated_data.time = self.retrieve_field("I", 0, 4)
        updated_data.paused = self.retrieve_field("I", 4, 8)

        updated_data.ets2_telemetry_plugin_revision = self.retrieve_field("I", 8, 12)
        updated_data.ets2_version_major = self.retrieve_field("I", 12, 16)
        updated_data.ets2_version_minor = self.retrieve_field("I", 16, 20)

        updated_data.flags = self.mmap[20:24]

        updated_data.speed = self.retrieve_field("f", 24, 28)
        updated_data.accelerationX = self.retrieve_field("f", 28, 32)
        updated_data.accelerationY = self.retrieve_field("f", 32, 36)
        updated_data.accelerationZ = self.retrieve_field("f", 36, 40)

        updated_data.coordinateX = self.retrieve_field("f", 40, 44)
        updated_data.coordinateY = self.retrieve_field("f", 44, 48)
        updated_data.coordinateZ = self.retrieve_field("f", 48, 52)

        updated_data.rotationX = self.retrieve_field("f", 52, 56)
        updated_data.rotationY = self.retrieve_field("f", 56, 60)
        updated_data.rotationZ = self.retrieve_field("f", 60, 64)

        updated_data.gear = self.retrieve_field("I", 64, 68)
        updated_data.gears = self.retrieve_field("I", 68, 72)
        updated_data.gearRanges = self.retrieve_field("I", 72, 76)
        updated_data.gearRangeActive = self.retrieve_field("I", 76, 80)

        updated_data.engineRpm = self.retrieve_field("f", 80, 84)
        updated_data.engineRpmMax = self.retrieve_field("f", 84, 88)

        updated_data.fuel = self.retrieve_field("f", 88, 92)
        updated_data.fuelCapacity = self.retrieve_field("f", 92, 96)
        updated_data.fuelRate = self.retrieve_field("f", 96, 100)
        updated_data.fuelAvgConsumption = self.retrieve_field("f", 100, 104)

        updated_data.userSteer = self.retrieve_field("f", 104, 108)
        updated_data.userThrottle = self.retrieve_field("f", 108, 112)
        updated_data.userBrake = self.retrieve_field("f", 112, 116)
        updated_data.userClutch = self.retrieve_field("f", 116, 120)

        updated_data.gameSteer = self.retrieve_field("f", 120, 124)
        updated_data.gameThrottle = self.retrieve_field("f", 124, 128)
        updated_data.gameBrake = self.retrieve_field("f", 128, 132)
        updated_data.gameClutch = self.retrieve_field("f", 132, 136)

        updated_data.truckWeight = self.retrieve_field("f", 136, 140)
        updated_data.trailerWeight = self.retrieve_field("f", 140, 144)

        updated_data.modelOffset = self.retrieve_field("I", 144, 148)
        updated_data.modelLength = self.retrieve_field("I", 148, 152)

        updated_data.trailerOffset = self.retrieve_field("I", 152, 156)
        updated_data.trailerLength = self.retrieve_field("I", 156, 160)

        updated_data.timeAbsolute = self.retrieve_field("I", 160, 164)
        updated_data.gearsReverse = self.retrieve_field("I", 164, 168)

        updated_data.trailerMass = self.retrieve_field("f", 168, 172)
        updated_data.trailerId = self.mmap[172: 236]
        updated_data.trailerName = self.mmap[236: 300]

        updated_data.jobIncome = self.retrieve_field("I", 300, 304)
        updated_data.jobDeadline = self.retrieve_field("I", 304, 308)
        updated_data.jobCitySource = self.mmap[308: 372]
        updated_data.jobCityDestination = self.mmap[372: 436]
        updated_data.jobCompanySource = self.mmap[436: 500]
        updated_data.jobCompanyDestination = self.mmap[500: 564]

        updated_data.retarderBrake = self.retrieve_field("I", 564, 568)
        updated_data.shifterSlot = self.retrieve_field("I", 568, 572)
        updated_data.shifterToggle = self.retrieve_field("I", 572, 576)
        updated_data.fill = self.retrieve_field("I", 576, 580)

        updated_data.aux = self.mmap[580:604]

        updated_data.airPressure = self.retrieve_field("f", 604, 608)
        updated_data.brakeTemperature = self.retrieve_field("f", 608, 612)

        updated_data.fuelWarning = self.retrieve_field("I", 612, 616)

        updated_data.adblue = self.retrieve_field("f", 616, 620)
        updated_data.adblueConsumption = self.retrieve_field("f", 620, 624)
        updated_data.oilPressure = self.retrieve_field("f", 624, 628)
        updated_data.oilTemperature = self.retrieve_field("f", 628, 632)
        updated_data.waterTemperature = self.retrieve_field("f", 632, 636)
        updated_data.batteryVoltage = self.retrieve_field("f", 636, 640)
        updated_data.lightsDashboard = self.retrieve_field("f", 640, 644)
        updated_data.wearEngine = self.retrieve_field("f", 644, 648)
        updated_data.wearTransmission = self.retrieve_field("f", 648, 652)
        updated_data.wearCabin = self.retrieve_field("f", 652, 656)
        updated_data.wearChassis = self.retrieve_field("f", 656, 660)
        updated_data.wearWheels = self.retrieve_field("f", 660, 664)
        updated_data.wearTrailer = self.retrieve_field("f", 664, 668)
        updated_data.truckOdometer = self.retrieve_field("f", 668, 672)
        updated_data.cruiseControlSpeed = self.retrieve_field("f", 672, 676)

        updated_data.truckMake = self.mmap[676:740]
        updated_data.truckMakeId = self.mmap[740:804]
        updated_data.truckModel = self.mmap[804:868]

        updated_data.speedlimit = self.retrieve_field("f", 868, 872)

        updated_data.routeDistance = self.retrieve_field("f", 872, 876)
        updated_data.routeTime = self.retrieve_field("f", 876, 880)

        updated_data.fuelRange = self.retrieve_field("f", 880, 884)

        updated_data.gearRatioForward = self.retrieve_array("24f", 884, 980)
        updated_data.gearRatioReverse = self.retrieve_array("8f", 980, 1012)
        updated_data.gearRatioDifferential = self.retrieve_field("f", 1012, 1016)

        updated_data.gearDashboard = self.retrieve_field("I", 1016, 1020)

        updated_data.onJob = self.retrieve_field("b", 1020, 1021)
        updated_data.jobFinished = self.retrieve_field("b", 1021, 1022)

        return updated_data

    def retrieve_field(self, f, start, end):
        field_data = struct.unpack(f, self.mmap[start:end])[0]
        return field_data

    def retrieve_array(self, f, start, end):
        array_data = struct.unpack(f, self.mmap[start:end])
        return array_data
