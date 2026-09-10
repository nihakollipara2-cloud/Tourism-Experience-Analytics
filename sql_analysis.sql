-- SQLite-compatible queries. Load tourism_cleaned.csv as table tourism.
SELECT Attraction, AttractionType, AVG(Rating) avg_rating, COUNT(*) ratings FROM tourism GROUP BY Attraction,AttractionType HAVING COUNT(*)>=20 ORDER BY avg_rating DESC,ratings DESC;
SELECT VisitModeName, COUNT(*) visits, ROUND(100.0*COUNT(*)/(SELECT COUNT(*) FROM tourism),2) percentage FROM tourism GROUP BY VisitModeName ORDER BY visits DESC;
SELECT AttractionType, ROUND(AVG(Rating),3) avg_rating, COUNT(*) visits FROM tourism GROUP BY AttractionType ORDER BY avg_rating DESC;
SELECT UserCountry, COUNT(*) visits FROM tourism GROUP BY UserCountry ORDER BY visits DESC LIMIT 10;
SELECT VisitYear, VisitMonthNum, COUNT(*) visits, ROUND(AVG(Rating),3) avg_rating FROM tourism GROUP BY VisitYear,VisitMonthNum ORDER BY VisitYear,VisitMonthNum;
SELECT VisitModeName, ROUND(AVG(Rating),3) avg_rating, COUNT(*) visits FROM tourism GROUP BY VisitModeName ORDER BY avg_rating DESC;
