from pandera import DataFrameModel, Field
from pandera.dtypes import Category
from pandera.typing import Series


class PowerConsumptionSchema(DataFrameModel):
    """Raw power consumption dataset schema."""

    Temperature: Series[float] = Field(nullable=True)
    Humidity: Series[float] = Field(nullable=True)
    Wind_Speed: Series[float] = Field(nullable=True)
    Hour: Series[int] = Field(nullable=True)
    Day: Series[int] = Field(nullable=True)
    Month: Series[int] = Field(nullable=True)
    general_diffuse_flows: Series[float] = Field(nullable=True)
    diffuse_flows: Series[float] = Field(nullable=True)
    DayOfWeek: Series[Category] = Field(nullable=True)
    IsWeekend: Series[Category] = Field(nullable=True)
    Zone_1_Power_Consumption: Series[float] = Field(nullable=True)
    Zone_2_Power_Consumption: Series[float] = Field(nullable=True)
    Zone_3_Power_Consumption: Series[float] = Field(nullable=True)

    class Config:
        coerce = True


class ProcessedPowerConsumptionSchema(DataFrameModel):
    """Processed power consumption dataset schema."""

    Temperature: Series[float] = Field(nullable=True)
    Humidity: Series[float] = Field(nullable=True)
    Wind_Speed: Series[float] = Field(nullable=True)
    Hour: Series[int] = Field(nullable=True)
    Day: Series[int] = Field(nullable=True)
    Month: Series[int] = Field(nullable=True)
    general_diffuse_flows: Series[float] = Field(nullable=True)
    diffuse_flows: Series[float] = Field(nullable=True)
    DayOfWeek_1: Series[Category] = Field(nullable=True)
    DayOfWeek_2: Series[Category] = Field(nullable=True)
    DayOfWeek_3: Series[Category] = Field(nullable=True)
    DayOfWeek_4: Series[Category] = Field(nullable=True)
    DayOfWeek_5: Series[Category] = Field(nullable=True)
    DayOfWeek_6: Series[Category] = Field(nullable=True)
    IsWeekend_1: Series[Category] = Field(nullable=True)
    Zone_1_Power_Consumption: Series[float] = Field(nullable=True)
    Zone_2_Power_Consumption: Series[float] = Field(nullable=True)
    Zone_3_Power_Consumption: Series[float] = Field(nullable=True)

    class Config:
        coerce = True
