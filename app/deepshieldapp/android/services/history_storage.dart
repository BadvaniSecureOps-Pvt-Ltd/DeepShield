import 'dart:convert';
import 'package:deepshield/models/scan_record.dart';
import 'package:shared_preferences/shared_preferences.dart';
import '../models/scan_record.dart';

class HistoryStorage {
  static const _key = 'deepshield_history';

  static Future<List<ScanRecord>> getAll() async {
    final sp = await SharedPreferences.getInstance();
    final raw = sp.getStringList(_key) ?? [];
    return raw.map((r) => ScanRecord.fromJson(jsonDecode(r))).toList();
  }

  static Future<void> addRecord(ScanRecord record) async {
    final sp = await SharedPreferences.getInstance();
    final raw = sp.getStringList(_key) ?? [];
    raw.insert(0, jsonEncode(record.toJson()));
    if (raw.length > 100) raw.removeRange(100, raw.length);
    await sp.setStringList(_key, raw);
  }

  static Future<void> clear() async {
    final sp = await SharedPreferences.getInstance();
    await sp.remove(_key);
  }
}
