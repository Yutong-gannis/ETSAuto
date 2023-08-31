from lib.ets2sdkdata import Ets2SdkData, Ets2SdkBoolean
from numpy import resize


class Ets2Telemetry:
    class _Version:
        def __init__(self, data: Ets2SdkData):
            self.SdkPlugin = data.ets2_telemetry_plugin_revision
            self.Ets2Major = data.ets2_version_major
            self.Ets2Minor = data.ets2_version_minor

    class _Physics:
        def __init__(self, data: Ets2SdkData):
            self.Speed = data.speed
            self.SpeedKmh = data.speed * 3.6
            self.SpeedMph = self.SpeedKmh / 1.6

            self.AccelerationX = data.accelerationX
            self.AccelerationY = data.accelerationY
            self.AccelerationZ = data.accelerationZ

            self.CoordinateX = data.coordinateX
            self.CoordinateY = data.coordinateY
            self.CoordinateZ = data.coordinateZ

            self.RotationX = data.rotationX
            self.RotationY = data.rotationY
            self.RotationZ = data.rotationZ

    class _DriveTrain:
        def __init__(self, data: Ets2SdkData):
            self.ElectricEnabled = data.get_boolean(Ets2SdkBoolean.ELECTRIC_ENABLED)
            self.EngineEnabled = data.get_boolean(Ets2SdkBoolean.ENGINE_ENABLED)

            self.EngineRpm = data.engineRpm
            self.EngineRpmMAX = data.engineRpmMax

            self.Gear = data.gear
            self.GearRange = data.gearRangeActive
            self.GearRanges = data.gearRanges
            self.GearsForward = data.gears
            self.GearsReverse = data.gearsReverse
            self.GearRatiosForward = resize(data.gearRatioForward, self.GearsForward)
            self.GearRatiosReverse = resize(data.gearRatioReverse, self.GearsReverse)
            self.GearRatioDifferential = data.gearRatioDifferential
            self.GearDashboard = data.gearDashboard

            self.Fuel = data.fuel
            self.FuelMax = data.fuelCapacity
            self.FuelRate = data.fuelRate
            self.FuelAvgConsumption = data.fuelAvgConsumption
            self.FuelRange = data.fuelRange
            self.FuelWarningLight = data.fuelWarning

            self.AirPressure = data.airPressure
            self.BrakeTemperature = data.brakeTemperature

            self.Adblue = data.adblue
            self.AdblueConsumption = data.adblueConsumption
            self.OilPressure = data.oilPressure
            self.OilTemperature = data.oilTemperature
            self.WaterTemperature = data.waterTemperature
            self.BatteryVoltage = data.batteryVoltage

            self.TruckOdometer = data.truckOdometer
            self.CruiseControl = data.get_boolean(Ets2SdkBoolean.CRUISE_CONTROL)
            self.CruiseControlSpeed = data.cruiseControlSpeed
            self.CruiseControlSpeedKmh = data.cruiseControlSpeed * 3.6
            self.CruiseControlSpeedMph = self.CruiseControlSpeedKmh / 1.6
            self.MotorBrake = data.get_boolean(Ets2SdkBoolean.MOTOR_BRAKE)
            self.ParkingBrake = data.get_boolean(Ets2SdkBoolean.PARK_BRAKE)
            self.Retarder = data.retarderBrake
            self.ShifterSlot = data.shifterSlot
            self.ShifterToggle = data.shifterToggle

    class _Controls:
        def __init__(self, data: Ets2SdkData):
            self.UserSteer = data.userSteer
            self.UserThrottle = data.userThrottle
            self.UserBrake = data.userBrake
            self.UserClutch = data.userClutch

            self.GameSteer = data.gameSteer
            self.GameThrottle = data.gameThrottle
            self.GameBrake = data.gameBrake
            self.GameClutch = data.gameClutch

    class _Job:
        def __init__(self, data: Ets2SdkData):
            self.OnJob = data.onJob != 0
            self.JobFinished = data.jobFinished != 0

            self.TrailerAttached = data.get_boolean(Ets2SdkBoolean.TRAILER_ATTACHED)
            self.Mass = data.trailerMass
            self.TrailerId = data.trailerId.decode("utf-8")
            self.TrailerName = data.trailerName.decode("utf-8")
            self.Cargo = self.TrailerName

            self.Income = data.jobIncome
            self.Deadline = data.jobDeadline
            self.NavigationDistanceLeft = data.routeDistance
            self.NavigationTimeLeft = data.routeTime

            self.CitySource = data.jobCitySource.decode("utf-8")
            self.CityDestination = data.jobCityDestination.decode("utf-8")
            self.CompanySource = data.jobCompanySource.decode("utf-8")
            self.CompanyDestination = data.jobCompanyDestination.decode("utf-8")

    class _Auxiliary:
        def __init__(self, data: Ets2SdkData):
            self.Wipers = data.get_boolean(Ets2SdkBoolean.WIPERS)

            self.BatteryVoltageWarning = data.get_boolean(Ets2SdkBoolean.BATTERY_VOLTAGE_WARNING)
            self.AirPressureWarning = data.get_boolean(Ets2SdkBoolean.AIR_PRESSURE_WARNING)
            self.AirPressureEmergency = data.get_boolean(Ets2SdkBoolean.AIR_PRESSURE_EMERGENCY)
            self.AdblueWarning = data.get_boolean(Ets2SdkBoolean.ADBLUE_WARNING)
            self.OilPressureWarning = data.get_boolean(Ets2SdkBoolean.OIL_PRESSURE_WARNING)
            self.WaterTemperatureWarning = data.get_boolean(Ets2SdkBoolean.WATER_TEMPERATURE_WARNING)

    class _Damage:
        def __init__(self, data: Ets2SdkData):
            self.WearEngine = data.wearEngine
            self.WearTransmission = data.wearTransmission
            self.WearCabin = data.wearCabin
            self.WearChassis = data.wearChassis
            self.WearWheels = data.wearWheels
            self.WearTrailer = data.wearTrailer

    class _Lights:
        def __init__(self, data: Ets2SdkData):
            self.BlinkerLeftActive = data.get_boolean(Ets2SdkBoolean.BLINKER_LEFT_ACTIVE)
            self.BlinkerRightActive = data.get_boolean(Ets2SdkBoolean.BLINKER_RIGHT_ACTIVE)
            self.BlinkerLeftOn = data.get_boolean(Ets2SdkBoolean.BLINKER_LEFT_ON)
            self.BlinkerRightOn = data.get_boolean(Ets2SdkBoolean.BLINKER_RIGHT_ON)

            self.ParkingLights = data.get_boolean(Ets2SdkBoolean.LIGHTS_PARKING)
            self.LowBeam = data.get_boolean(Ets2SdkBoolean.LIGHTS_BEAM_LOW)
            self.HighBeam = data.get_boolean(Ets2SdkBoolean.LIGHTS_BEAM_HIGH)
            self.FrontAux = data.get_boolean(Ets2SdkBoolean.LIGHTS_AUX_FRONT)
            self.RoofAux = data.get_boolean(Ets2SdkBoolean.LIGHTS_AUX_ROOF)
            self.Beacon = data.get_boolean(Ets2SdkBoolean.LIGHTS_BEACON)
            self.BrakeLights = data.get_boolean(Ets2SdkBoolean.LIGHTS_BRAKE)
            self.ReverseLights = data.get_boolean(Ets2SdkBoolean.LIGHTS_REVERSE)
            self.DashboardLights = data.lightsDashboard

    def __init__(self, data: Ets2SdkData):
        self.Time = data.time
        self.Pause = data.paused > 0

        self.TruckId = data.truckModel
        self.Truck = data.truckModel.decode("utf-8")
        self.Manufacturer = data.truckMake.decode("utf-8")
        self.ManufacturerId = data.truckMakeId.decode("utf-8")

        self.Version = self._Version(data)
        self.Physics = self._Physics(data)
        self.DriveTrain = self._DriveTrain(data)
        self.Controls = self._Controls(data)
        self.Job = self._Job(data)
        self.Auxiliary = self._Auxiliary(data)
        self.Damage = self._Damage(data)
        self.Lights = self._Lights(data)
