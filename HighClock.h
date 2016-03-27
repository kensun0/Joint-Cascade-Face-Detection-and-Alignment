#include <windows.h> 

LARGE_INTEGER TimerStart();
LARGE_INTEGER TimerEnd();
LARGE_INTEGER GetTime(LARGE_INTEGER start,LARGE_INTEGER end);


LARGE_INTEGER TimerStart()
{
	LARGE_INTEGER start; 
	QueryPerformanceCounter(&start);
	return start;
}

LARGE_INTEGER TimerEnd()
{
	LARGE_INTEGER end; 
	QueryPerformanceCounter(&end);
	return end;
}

LARGE_INTEGER GetTime(LARGE_INTEGER start,LARGE_INTEGER end)
{
	LARGE_INTEGER Freq;
	QueryPerformanceFrequency(&Freq);
	LARGE_INTEGER use;
	use.QuadPart=( end.QuadPart - start.QuadPart)*1000/ Freq.QuadPart;
	return use;
}

//! High-resolution clock (In fraction of seconds, as milliseconds)
//单位:秒
class HighClock
{
public:
	
	HighClock()
	{
		// Setup the high-resolution timer
		QueryPerformanceFrequency(&TicksPerSec);
		
		// High-resolution timers, iniitalized timer
		QueryPerformanceCounter(&startTime);
		
		// Start off the timer
		Start(); Stop();
	}
	
	void Stop()
	{
		QueryPerformanceCounter(&endTime);
	}
	
	// In seconds
	float GetTime()
	{
		// Measure the cycle time
		cpuTime.QuadPart = endTime.QuadPart - startTime.QuadPart;
		return (float)cpuTime.QuadPart / (float)TicksPerSec.QuadPart;
	}
	
	void Start()
	{
		QueryPerformanceCounter(&startTime);
	}
	
private:
	LARGE_INTEGER TicksPerSec;
	LARGE_INTEGER startTime, endTime, cpuTime;
};
