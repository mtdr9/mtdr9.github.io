---
layout: post
title: What Worked at Wayfair
subtitle: And how to build a money printing machine
image: /img/DALL·E 2023-02-14 18.04.41 - a dutch masters painting of gold coins raining from the sky.png
tags: [management, analytics, professional, development]
---

In four years as a search marketer, I built - with the help of my team of analysts and our partners in data science, ad tech, and storefront - $87 million of annualized, incremental revenue for Wayfair. This post isn't about any specific project, but rather the broad themes that allowed me to make the right bets and execute them quickly, as well as some of the lessons learned from bets that didn't pay off. I've wrapped it up into three rules for success. Now, my uncle once recommended me a book of 11 rules for business, none of which are featured here, so your mileage may vary.

1. Define what's important and what's not
2. Manage your analysts to their skill level
3. Be your own coach

**1. Define what's important and what's not**

From a single 90 minute focus session to a 2-year roadmap, knowing what to focus on makes it so much more likely that by the quarterly performance slide, you'll have achieved some substantial, measureable results to write in. And it'll even let you manage upwards, telling your boss "I can get that done, but I think we should focus on A and B first, for reason X".

My key insight here is that different prioritization techniques work best for different time horizons. Here are explanations of the 3 techniques I'll be discussing, courtesy of [ChatGPT](https://platform.openai.com/playground):

RICE prioritization
> The RICE prioritization framework is a popular tool used by many product managers and project managers. It stands for Risk, Impact, Confidence, and Effort, and it is used to prioritize tasks or projects based on four key factors.
> 1. Risk: This factor measures the potential risks associated with the task or project. For example, a project that involves developing a new product has a higher risk than a project that involves minor improvements to an existing product.
> 2. Impact: This factor measures the potential impact of the task or project. For example, a project that has the potential to significantly increase revenue has a higher impact than a project that has a minor impact on revenue.
> 3. Confidence: This factor measures the level of confidence that the team has in the task or project. For example, a project that has a well-defined plan and clear objectives has a higher confidence level than a project that lacks clarity and structure.
> 4. Effort: This factor measures the amount of effort required to complete the task or project. For example, a project that requires a significant amount of time and resources has a higher effort level than a project that requires minimal effort.  

> To use the RICE prioritization framework, assign a score to each of these four factors, and then multiply them together to get a final score. For example, if a project has a risk score of 8, an impact score of 7, a confidence score of 9, and an effort score of 5, its RICE score would be 2520 (8 x 7 x 9 x 5).

Effort vs. Value
> The effort vs. value framework is a simple and effective way to prioritize tasks or projects based on their potential value and the effort required to complete them.
> 1. Effort: This factor measures the amount of effort required to complete the task or project. For example, a project that requires a significant amount of time and resources has a higher effort level than a project that requires minimal effort.
> 2. Value: This factor measures the potential value of the task or project. For example, a project that has the potential to significantly increase revenue has a higher value than a project that has a minor impact on revenue.  

> To use the effort vs. value framework, plot each task or project on a graph with effort on the x-axis and value on the y-axis. Tasks or projects that require high effort but have low value should be deprioritized, while tasks or projects that have high value and low effort should be prioritized.

Urgency vs. Importance
> The urgency vs. importance framework is a classic tool used to prioritize tasks or projects based on their urgency and importance.
> 1. Urgency: This factor measures how quickly the task or project needs to be completed. For example, a task that has a tight deadline has a higher urgency than a task with a more relaxed deadline.
> 2. Importance: This factor measures the importance of the task or project. For example, a task that is critical to the success of a project has a higher importance than a task that is not critical.  

> To use the urgency vs. importance framework, plot each task or project on a matrix with urgency on the x-axis and importance on the y-axis. Tasks or projects that are both urgent and important should be prioritized, while tasks or projects that are not urgent or important should be deprioritized.

To determine what to do for the next 6-12 months, I'd start with a brainstorming session, getting all of the ideas from the team on paper, without judging their quality. Then, I'd quickly sort ideas on a 2x2 matrix of effort and value. There's probably a lot of ideas, and I want to pluck the low hanging fruit - those high value, low effort ideas - as quickly as possible. Once I have a set of "viable" ideas from the effort vs. value exercise, I'd run them through full RICE prioritization to figure out the relative quality of each, which I'd use to construct my roadmap.

To determine what to do right now - today, or in my next 90-minute focus session - I'd look at my candidate projects' urgency vs. importance. Highly urgent AND highly urgent? You'd better finish this prioritization exercise quickly because that work won't do itself. Not important? Why is it on your list? Can it be eliminated, reduced, or snoozed? 

Personally, I sometimes find it's hard to quickly say no to things with at least some value, especially if there's a stakeholder on the other side. For this reason I also keep a bucket on my to do list of "probably never doing". This lets me demarcate that I probably will never have time to prioritize something, without my to-do list falling prey to the [endowment effect](https://en.wikipedia.org/wiki/Endowment_effect). I can also keep it top of mind for a bit while I process whether it's really worth taking off the list. Just note that if something goes on the "probably never doing" list, it's usually smart to proactively get alignment with stakeholders!

**2. Manage your analysts to their skill level**

..........give analysts work that stretches them, but that they can achieve. So depending on how much they're streched, that's how much you need to coach them

Foundational to succeeding as a manager is being able to "scope" large, ambiguous problems: clarifying their outputs, and breaking down the steps to get there into more concrete pieces. An example problem statement could be: "every customer who clicks on a product listing ad gets sent to the same landing page template. How do we change the landing page to create more revenue"? You can't just give that problem to a day 1 new hire out of school (i.e. a level 1 or L1): chances are, they'll spend a long time on it, and perhaps produce a recommendation, but with weak evidence that it's the highest effort, lowest value option possible.

.......... instead you need to define the goal (grow revenue, or save costs and reinvest them), and create a project structure. 1. create a framework for what drives value on the page based on the different components of the page; 2. run analysis; 3. launch an [MVP](https://jexo.io/blog/ppm-glossary-what-is-mvp/) test to confirm one of the hypotheses; 4. if successful, scale it. Then assess: is it scoped enough for an analyst to run? A strong senior analyst could probably succeed here, with the right coach. And that brings us to situational leadership.

> Situational leadership is a management style that recognizes and responds to the needs of different employees in different situations. It is a flexible approach that can be used to motivate, empower, and challenge employees to reach their goals. Based on the needs of each individual or team for a given task, situational leadership calls for one of four approaches:
> 1. Directing is most appropriate when an employee is new to a task or organization. It involves the leader providing clear instructions and expectations and closely monitoring performance. The leader should outline step-by-step what they want the employee to do and provide guidance as needed.
> 2. Coaching is an appropriate strategy when an employee has some experience but is still in the process of learning. It involves the leader providing guidance, feedback, and support while allowing the employee to take the lead. The leader should ask questions, provide feedback, and offer assistance when needed.
> 3. Supporting is most appropriate when an employee is fairly experienced and able to work independently. It involves the leader providing encouragement and guidance while allowing the employee to take the lead. The leader should provide resources, challenge the employee to take on more responsibility, and offer assistance when needed.
> 4. Delegating is an appropriate strategy when an employee is highly experienced and capable of handling the task without much guidance. It involves the leader assigning tasks to the employee and trusting that they will complete the task on their own. The leader should provide resources, set expectations, and offer support when needed.  
> 
> Situational leadership also requires regular communication between the leader and team members. Leaders should use regular check-ins to assess progress and provide feedback. This will help ensure that everyone is on the same page and that any issues are addressed in a timely manner. By taking the time to understand the situation and the individuals involved, leaders can use situational leadership to create an environment where employees feel empowered and motivated.
