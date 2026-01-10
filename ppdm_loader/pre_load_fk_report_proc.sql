CREATE OR ALTER PROCEDURE stg.usp_preload_fk_report
  @child_schema sysname,
  @child_table  sysname,
  @src_view     sysname,   -- e.g. 'dbo.stg_v_norm_dbo_well'
  @top_n        int = 50
AS
BEGIN
  SET NOCOUNT ON;

  DECLARE @child_obj int =
    OBJECT_ID(QUOTENAME(@child_schema) + '.' + QUOTENAME(@child_table));

  IF @child_obj IS NULL
  BEGIN
    RAISERROR('Child table not found', 16, 1);
    RETURN;
  END

  IF OBJECT_ID(@src_view) IS NULL
  BEGIN
    RAISERROR('Source view not found', 16, 1);
    RETURN;
  END

  ;WITH fkcols AS (
      SELECT
        fk.name AS fk_name,
        sch_child.name AS child_schema,
        tab_child.name AS child_table,
        col_child.name AS child_col,
        sch_parent.name AS parent_schema,
        tab_parent.name AS parent_table,
        col_parent.name AS parent_col,
        fkc.constraint_column_id
      FROM sys.foreign_keys fk
      JOIN sys.foreign_key_columns fkc
        ON fkc.constraint_object_id = fk.object_id
      JOIN sys.tables tab_child
        ON tab_child.object_id = fk.parent_object_id
      JOIN sys.schemas sch_child
        ON sch_child.schema_id = tab_child.schema_id
      JOIN sys.columns col_child
        ON col_child.object_id = tab_child.object_id
       AND col_child.column_id = fkc.parent_column_id
      JOIN sys.tables tab_parent
        ON tab_parent.object_id = fk.referenced_object_id
      JOIN sys.schemas sch_parent
        ON sch_parent.schema_id = tab_parent.schema_id
      JOIN sys.columns col_parent
        ON col_parent.object_id = tab_parent.object_id
       AND col_parent.column_id = fkc.referenced_column_id
      WHERE fk.parent_object_id = @child_obj
  ),
  fkgroups AS (
      SELECT
        fk_name,
        MIN(parent_schema) AS parent_schema,
        MIN(parent_table) AS parent_table,
        STRING_AGG(child_col, ',')  WITHIN GROUP (ORDER BY constraint_column_id) AS child_cols_csv,
        STRING_AGG(parent_col, ',') WITHIN GROUP (ORDER BY constraint_column_id) AS parent_cols_csv
      FROM fkcols
      GROUP BY fk_name
  )
  SELECT
    g.fk_name,
    @child_schema AS child_schema,
    @child_table  AS child_table,
    x.child_col,
    x.parent_schema,
    x.parent_table,
    x.parent_col,
    x.missing_count,
    x.sample_missing
  FROM fkgroups g
  CROSS APPLY (
      -- Build dynamic SQL per FK constraint so composite keys work
      SELECT
        CAST(NULL AS nvarchar(max)) AS child_col,
        CAST(g.parent_schema AS sysname) AS parent_schema,
        CAST(g.parent_table  AS sysname) AS parent_table,
        CAST(NULL AS nvarchar(max)) AS parent_col,
        CAST(0 AS bigint) AS missing_count,
        CAST(NULL AS nvarchar(max)) AS sample_missing
  ) x
  OPTION (RECOMPILE);

  -- NOTE:
  -- This skeleton returns the FK list; the next step is to run dynamic SQL per FK
  -- to compute missing_count/sample_missing and return the final rows.
  --
  -- If you want, I’ll give you the full dynamic portion, but the key fix is:
  --   iterate FK CONSTRAINTS, not columns, and do not depend on __NAT columns.
END
GO
