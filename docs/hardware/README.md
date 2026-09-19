# Machine register

Hardware for every machine in the group that produces training runs.
Run rows carry a `machine_id` rather than repeating these columns.

## Columns

| column | unit / type | source |
|---|---|---|
| `machine_id` | string | assigned by us, stable forever |
| `owner` | string | GitHub handle |
| `cpu_model` | string | `Win32_Processor.Name` |
| `cpu_cores_physical` | integer | `Win32_Processor.NumberOfCores` |
| `cpu_cores_logical` | integer | `Win32_Processor.NumberOfLogicalProcessors` |
| `cpu_base_clock_mhz` | integer, MHz | `Win32_Processor.MaxClockSpeed` |
| `ram_total_bytes` | integer, **bytes** | `Win32_ComputerSystem.TotalPhysicalMemory` |
| `gpu_model` | string | `Win32_VideoController.Name` |
| `gpu_dedicated` | boolean | judgement — discrete card or integrated |
| `gpu_vram_bytes` | integer, bytes, nullable | see caveat 2 |
| `os_name` | string | `Win32_OperatingSystem.Caption` |
| `os_version` | string | `Win32_OperatingSystem.Version` |
| `os_build` | string | `Win32_OperatingSystem.BuildNumber` |
| `notes` | free text | anything that would mislead a reader |

## Caveats

**1. `cpu_base_clock_mhz` is a nameplate figure, not a measurement.**
`MaxClockSpeed` reports the rated base frequency. Modern CPUs boost well
above it under sustained load, and how far depends on cooling, power
profile and how many cores are busy. It is a label for the chip, not a
description of what happened during a training run.

**2. `gpu_vram_bytes` cannot be read reliably from WMI.**
`Win32_VideoController.AdapterRAM` is a 32-bit field that cannot represent
modern VRAM. On m01 it returns 2147479552 (~2 GiB) for a 16 GB device.
Take VRAM from the vendor tool or the device name, and record how you got it.

**3. `ram_total_bytes` is memory visible to the OS, not installed memory.**
`TotalPhysicalMemory` excludes firmware- and hardware-reserved regions.

## Collecting a new machine

In PowerShell:

```powershell
Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors,MaxClockSpeed
Get-CimInstance Win32_ComputerSystem | Select-Object TotalPhysicalMemory
Get-CimInstance Win32_VideoController | Select-Object Name,AdapterRAM
Get-CimInstance Win32_OperatingSystem | Select-Object Caption,Version,BuildNumber
```