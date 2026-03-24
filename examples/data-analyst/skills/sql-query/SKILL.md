---
name: sql-query
description: >
  Execute read-only SQL queries against a SQLite database and return structured
  results as JSON. Use when the user asks to query, filter, aggregate, join, or
  explore tabular data. Supports SELECT, WITH (CTEs), EXPLAIN, and PRAGMA.
---

# SQL Query Skill

Executes read-only SQL queries against a SQLite database and returns results
as a JSON object with `row_count`, `columns`, and `rows`.

## Usage notes

- SELECT / WITH / EXPLAIN / PRAGMA only — INSERT, UPDATE, DELETE, DROP are blocked
- Results are a JSON array of row objects; column names come from the query
- NULL values are returned as JSON `null`
- Default database: `./data/chinook.db` (Chinook digital media store)

## Chinook schema reference

Table names are **CamelCase** — use them exactly as shown.

| Table | Columns |
|---|---|
| `Invoice` | InvoiceId, CustomerId, InvoiceDate, BillingAddress, BillingCity, BillingState, BillingCountry, BillingPostalCode, Total |
| `InvoiceLine` | InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity |
| `Track` | TrackId, Name, AlbumId, MediaTypeId, GenreId, Composer, Milliseconds, Bytes, UnitPrice |
| `Genre` | GenreId, Name |
| `Customer` | CustomerId, FirstName, LastName, Company, Address, City, State, Country, PostalCode, Phone, Fax, Email, SupportRepId |
| `Album` | AlbumId, Title, ArtistId |
| `Artist` | ArtistId, Name |
| `Employee` | EmployeeId, LastName, FirstName, Title, ReportsTo, BirthDate, HireDate, Address, City, State, Country, PostalCode, Phone, Fax, Email |
| `MediaType` | MediaTypeId, Name |
| `Playlist` | PlaylistId, Name |
| `PlaylistTrack` | PlaylistId, TrackId |

## Example queries

```sql
-- Revenue by country (top markets)
SELECT BillingCountry, ROUND(SUM(Total), 2) AS Revenue, COUNT(*) AS Orders
FROM Invoice
GROUP BY BillingCountry
ORDER BY Revenue DESC
LIMIT 10;

-- Revenue by genre
SELECT g.Name AS Genre, ROUND(SUM(il.UnitPrice * il.Quantity), 2) AS Revenue
FROM InvoiceLine il
JOIN Track t ON il.TrackId = t.TrackId
JOIN Genre g ON t.GenreId = g.GenreId
GROUP BY g.Name
ORDER BY Revenue DESC;

-- Monthly revenue trend
SELECT strftime('%Y-%m', InvoiceDate) AS Month, ROUND(SUM(Total), 2) AS Revenue
FROM Invoice
GROUP BY Month
ORDER BY Month;

-- Genre performance by top market
SELECT i.BillingCountry, g.Name AS Genre,
       ROUND(SUM(il.UnitPrice * il.Quantity), 2) AS Revenue
FROM Invoice i
JOIN InvoiceLine il ON i.InvoiceId = il.InvoiceId
JOIN Track t ON il.TrackId = t.TrackId
JOIN Genre g ON t.GenreId = g.GenreId
WHERE i.BillingCountry IN ('USA', 'Canada', 'France', 'Brazil', 'Germany')
GROUP BY i.BillingCountry, g.Name
ORDER BY i.BillingCountry, Revenue DESC;
```
