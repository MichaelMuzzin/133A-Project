


<h3 style="color:#7216e6">Goal of this project:</h3>
<p>The goal of this project is to utilize the topics we learned in ECE 133A to a real-world scenario of algorithms, analyzing data, computing. This includes k-means clustering and singular value decomposition amongst other topics to predict the income of an individual based on current data regarding major, income, company, etc.
Using a 2024 Stackoverflow survey with more that 70,000 responses, we able to compile a comprehensive and accurate set of data inputs. In order to make the computations easier. We narrowed the data to just United States participants and removed some questions that we deemed unnecessary to our analysis. As with the project guidlines, there are 4 individual parts to this project that we have outlined: PART 1-4

<h3 style="color:#7216e6">Students in this group:</h3>
**Final Summary of Part 1:**
-  **Shahab Besharatlou**
-  **Michael Muzzin**
-  **Sina Ghadimi**


<h3 style="color:#7216e6">PART 1:</h3>
<p>**PART 1 ANALYSIS:**

<h4 style="color:#FF5733">Most Popular Programming Languages in the U.S. (2024)</h4>


Chart Type: Horizontal Bar Chart

This chart displays the 20 most commonly used programming languages among U.S. developers in the dataset.

How It Was Generated:

* The “LanguageHaveWorkedWith” column was extracted.  
* Entries were split by semicolons to count occurrences of each language.  
* The top 20 languages were plotted in a horizontal bar chart with labels inside the bars.

Key Insights:

* The most popular languages include Python, JavaScript, SQL, and Java, indicating their widespread use in the industry.  
* The chart helps identify which programming languages are most valuable in the job market.

<h4 style="color:#FF5733">Salary Data by Age Groups</h4>


Chart Type: Bar Chart

A series of bar charts were created to compare salary distributions across different age groups.

How It Was Generated:

* Developers were grouped into age categories:  
  * Under 18  
  * 18-24 years old  
  * 25-34 years old  
  * 35-44 years old  
  * 55-64 years old  
  * Salary statistics for each group were calculated, including:  
  * Maximum salary  
  * Minimum salary  
  * Median salary  
  * Mean salary  
  * Standard deviation  
  * Each category was plotted with a separate bar chart.

Key Insights:

* Income increases with age up to a certain point, with developers aged 25-44 earning the highest salaries.  
* Young developers (18-24 years old) earn significantly lower salaries compared to those over 30\.  
* A decline in salaries after age 55 suggests that fewer respondents in this age group report high compensation.

<h4 style="color:#FF5733">3 Developer Roles & Popularity</h4>


Chart Type: Horizontal Bar Chart

This chart shows the most common job titles in the dataset.

How It Was Generated:

* The “DevType” column was used to count the number of respondents in each job category.  
* Job categories with higher representation were plotted in a horizontal bar chart.

Key Insights:

* The most common roles in the dataset include Software Developers, Web Developers, and Data Scientists.  
* Some specialized roles (e.g., Embedded Systems Engineers, DevOps Specialists) have fewer respondents.  
* Understanding role distribution helps tailor salary predictions for different job categories.

<h4 style="color:#FF5733">Salary Comparison by Career Path</h4>


Chart Type: Bar Chart

A separate bar chart was generated for each career group, showing how salaries vary based on career paths.

How It Was Generated:

* Developers were grouped based on career type.  
* Within each career group, salary metrics were calculated (max, min, median, mean, standard deviation).  
* These statistics were plotted as a bar chart.

Key Insights:

* Certain careers (e.g., Machine Learning, Cloud Computing) have significantly higher mean salaries than general software development roles.  
* Developers in low-paying fields (e.g., Technical Support, QA Testing) tend to have lower max salaries than other groups.  
* Median salaries offer a better representation than mean salaries due to the presence of outliers in some careers.

<h4 style="color:#FF5733">Salary Trends by Experience Level</h4>


Chart Type: Bar Chart

This chart visualizes salary trends based on years of experience.

How It Was Generated:

* Developers were categorized into experience levels (e.g., Entry-Level, Mid-Level, Senior).  
* Salary statistics for each level were aggregated and plotted in a bar chart.

Key Insights:

* Salary grows exponentially with experience, with senior developers earning significantly higher wages.  
* Entry-level positions have a wide salary range, indicating that experience isn’t the only factor affecting pay.  
* Some high-paying roles (e.g., AI Engineering, Cloud Architecture) require substantial experience.

<h4 style="color:#FF5733">Salary vs. Most Used Programming Languages</h4>


Chart Type: Scatter Plot

This chart explores the relationship between programming language usage and salary.

How It Was Generated:

* Developers using certain languages were grouped.  
* The average salary of developers using each language was calculated.  
* A scatter plot was generated to show how salaries vary by language.

Key Insights:

* C++, Rust, and Go developers tend to have higher average salaries than Python or JavaScript users.  
* Web development languages (e.g., JavaScript, PHP) are associated with lower salaries, likely due to lower barriers to entry.  
* Specialized languages (e.g., Swift, Kotlin, R) show more variation in salary distribution.

<h4 style="color:#FF5733">Income Distribution in the U.S.</h4>


Chart Type: Histogram

A histogram was created to visualize the distribution of salaries among U.S. developers.

How It Was Generated:

* The “CompTotal” column was used to analyze salary distribution.  
* Salaries above `$1,000,000` were excluded to remove outliers.  
* A histogram was plotted to show the frequency of salary ranges.

Key Insights:

* Most developers earn between `$50,000-$150,000`.  
* There is a long tail of high salaries, but few earn above `$500,000`.  
* The histogram suggests a log-normal distribution, meaning salary transformations (log-scale) are beneficial for predictive modeling.

<h4 style="color:#FF5733">Final Summary of Part 1:</h4>


1. Programming Languages: Python and JavaScript are the most common, but C++, Rust, and Go developers tend to earn more.  
2. Age & Salary: Salaries increase with age up to 45, then decline slightly.  
3. Job Titles: Software Developers and Data Scientists are the most common roles.  
4. Career Paths: AI, Cloud, and Machine Learning roles pay significantly higher.  
5. Experience & Salary: More experience \= higher pay, but some senior roles vary widely.  
6. Language vs. Salary: Backend and specialized languages tend to have higher salaries.  
7. Income Distribution: Most salaries fall in `$50,000-$150,000`, but some outliers exist.



<h3 style="color:#7216e6">PART 2:</h3>
<p>

<h3 style="color:#7216e6">PART 3:</h3>
<p>

<h3 style="color:#7216e6">PART 4:</h3>
