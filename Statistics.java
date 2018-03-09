package GitHubSample;

import java.util.Arrays;

// Source (need to confirm):
// http://stackoverflow.com/questions/7988486/how-do-you-calculate-the-variance-median-and-standard-deviation-in-c-or-java/7988556
public class Statistics {
    double[] data;
    int size;   

    public Statistics(double[] data) 
    {
        this.data = data;
        size = data.length;
    }   
    
	// changed the properties of these methods to public so that they will be
	// accessed in aim two experiments
    public double getMean()
    {
        double sum = 0.0;
        for(double a : data)
            sum += a;
        return sum/size;
    }

    public double getVariance()
    {
        double mean = getMean();
        double temp = 0;
        for(double a :data)
            temp += (mean-a)*(mean-a);
        return temp/size;
    }

    public double getStdDev()
    {
        return Math.sqrt(getVariance());
    }

    public double median() 
    {
       Arrays.sort(data);

       if (data.length % 2 == 0) 
       {
          return (data[(data.length / 2) - 1] + data[data.length / 2]) / 2.0;
       } 
       else 
       {
          return data[data.length / 2];
       }
    }
}