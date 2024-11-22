using System;
using System.Diagnostics;

class Program
{
    static void Main()
    {
        ProcessStartInfo start = new ProcessStartInfo();
        start.FileName = "python";
        start.Arguments = "script.py";
        start.UseShellExecute = false;
        start.RedirectStandardOutput = true;
        start.RedirectStandardInput = true;

        using (Process process = Process.Start(start))
        {
            using (StreamWriter sw = process.StandardInput)
            {
                if (sw.BaseStream.CanWrite)
                {
                    sw.WriteLine("Hello from C#");
                }
            }

            using (StreamReader sr = process.StandardOutput)
            {
                string result = sr.ReadToEnd();
                Console.WriteLine(result);
            }
        }
    }
}
