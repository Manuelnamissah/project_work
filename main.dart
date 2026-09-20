import 'dart:convert';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:image_picker/image_picker.dart';
import 'package:shared_preferences/shared_preferences.dart';

void main() {
  runApp(const BreastCancerApp());
}

class BreastCancerApp extends StatelessWidget {
  const BreastCancerApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Breast Cancer Detection',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFFD84C83),
          brightness: Brightness.light,
        ),
        useMaterial3: true,
        scaffoldBackgroundColor: const Color(0xFFFFF7FA),
        appBarTheme: const AppBarTheme(
          backgroundColor: Color(0xFFFFF7FA),
          foregroundColor: Color(0xFF8E1D50),
          surfaceTintColor: Colors.transparent,
        ),
      ),
      home: const HomeScreen(),
    );
  }
}

class ScanHistoryItem {
  final String imagePath;
  final String resultType;
  final String prediction;
  final double confidence;
  final String filename;
  final DateTime time;
  final double benignScore;
  final double invalidScore;
  final double malignantScore;
  final String note;
  final String message;

  ScanHistoryItem({
    required this.imagePath,
    required this.resultType,
    required this.prediction,
    required this.confidence,
    required this.filename,
    required this.time,
    required this.benignScore,
    required this.invalidScore,
    required this.malignantScore,
    required this.note,
    required this.message,
  });

  Map<String, dynamic> toJson() {
    return {
      'imagePath': imagePath,
      'resultType': resultType,
      'prediction': prediction,
      'confidence': confidence,
      'filename': filename,
      'time': time.toIso8601String(),
      'benignScore': benignScore,
      'invalidScore': invalidScore,
      'malignantScore': malignantScore,
      'note': note,
      'message': message,
    };
  }

  factory ScanHistoryItem.fromJson(Map<String, dynamic> json) {
    return ScanHistoryItem(
      imagePath: json['imagePath']?.toString() ?? '',
      resultType: json['resultType']?.toString() ?? 'ok',
      prediction: json['prediction']?.toString() ?? '',
      confidence: (json['confidence'] as num?)?.toDouble() ?? 0.0,
      filename: json['filename']?.toString() ?? '',
      time: DateTime.tryParse(json['time']?.toString() ?? '') ?? DateTime.now(),
      benignScore: (json['benignScore'] as num?)?.toDouble() ?? 0.0,
      invalidScore: (json['invalidScore'] as num?)?.toDouble() ?? 0.0,
      malignantScore: (json['malignantScore'] as num?)?.toDouble() ?? 0.0,
      note: json['note']?.toString() ?? '',
      message: json['message']?.toString() ?? '',
    );
  }
}

class PredictionResult {
  final bool success;
  final String status;
  final String filename;
  final String prediction;
  final double confidence;
  final double benignScore;
  final double invalidScore;
  final double malignantScore;
  final String note;
  final String message;
  final String error;

  PredictionResult({
    required this.success,
    required this.status,
    required this.filename,
    required this.prediction,
    required this.confidence,
    required this.benignScore,
    required this.invalidScore,
    required this.malignantScore,
    required this.note,
    required this.message,
    required this.error,
  });

  factory PredictionResult.fromJson(Map<String, dynamic> json) {
    double parseNum(dynamic value) {
      if (value is num) return value.toDouble();
      if (value is String) return double.tryParse(value) ?? 0.0;
      return 0.0;
    }

    final prediction = json['prediction']?.toString().toLowerCase() ?? '';
    final explicitStatus = json['status']?.toString().toLowerCase();
    final rawScores = json['raw_scores'];
    final scoreMap = rawScores is Map
        ? Map<String, dynamic>.from(rawScores)
        : <String, dynamic>{};

    final rawScore = parseNum(json['raw_score']).clamp(0.0, 1.0).toDouble();

    // The current MobileNetV2 model has one sigmoid output:
    // raw_score = malignant probability, while 1 - raw_score = benign probability.
    double benign = parseNum(scoreMap['benign']);
    double malignant = parseNum(scoreMap['malignant']);
    double invalid = parseNum(scoreMap['invalid']);

    if (benign == 0.0 && malignant == 0.0) {
      benign = prediction == 'benign' && json['confidence'] != null
          ? parseNum(json['confidence'])
          : 1.0 - rawScore;
      malignant = prediction == 'malignant' && json['confidence'] != null
          ? parseNum(json['confidence'])
          : rawScore;
    }

    final String status;
    if (prediction == 'invalid' || explicitStatus == 'invalid') {
      status = 'invalid';
    } else if (explicitStatus == 'uncertain' || prediction == 'uncertain') {
      status = 'uncertain';
    } else if (prediction == 'benign' || prediction == 'malignant') {
      status = 'ok';
    } else {
      status = 'error';
    }

    return PredictionResult(
      success: json['success'] == true,
      status: status,
      filename: json['filename']?.toString() ?? '',
      prediction: prediction,
      confidence: parseNum(json['confidence']),
      benignScore: benign.clamp(0.0, 1.0).toDouble(),
      invalidScore: invalid.clamp(0.0, 1.0).toDouble(),
      malignantScore: malignant.clamp(0.0, 1.0).toDouble(),
      note: json['note']?.toString() ?? '',
      message: json['message']?.toString() ?? '',
      error: json['error']?.toString() ?? '',
    );
  }
}

class ApiService {
  // Deployed FastAPI backend on Render.
  static const String baseUrl = 'https://group-8d-5mz1.onrender.com';
  static const String predictUrl = '$baseUrl/predict';

  static Future<PredictionResult> predictImage(File imageFile) async {
    if (!await imageFile.exists()) {
      throw Exception('The selected image could not be found on the device.');
    }

    final uri = Uri.parse(predictUrl);
    final request = http.MultipartRequest('POST', uri);
    request.headers['Accept'] = 'application/json';

    final filename = imageFile.path.split(RegExp(r'[\\/]')).last;
    request.files.add(
      await http.MultipartFile.fromPath(
        'file',
        imageFile.path,
        filename: filename,
      ),
    );

    // Render services can take a little time to wake up after being idle.
    final streamedResponse = await request.send().timeout(
      const Duration(seconds: 180),
      onTimeout: () => throw Exception(
        'The backend took too long to respond. Please try the request again.',
      ),
    );

    final response = await http.Response.fromStream(streamedResponse);

    debugPrint('PREDICT STATUS: ${response.statusCode}');
    debugPrint('PREDICT BODY: ${response.body}');

    Map<String, dynamic> data;
    try {
      final decoded = jsonDecode(response.body);
      if (decoded is! Map) {
        throw const FormatException('Expected a JSON object.');
      }
      data = Map<String, dynamic>.from(decoded);
    } catch (_) {
      throw Exception(
        'The backend returned an unreadable response (${response.statusCode}).',
      );
    }

    // The backend returns a normal prediction with HTTP 200.
    // Invalid files are returned as HTTP 400/413 with prediction: invalid.
    if (response.statusCode == 200 ||
        data['prediction']?.toString().toLowerCase() == 'invalid') {
      return PredictionResult.fromJson(data);
    }

    final detail = data['detail']?.toString();
    final message = data['message']?.toString();
    throw Exception(
      'Server error ${response.statusCode}: ${detail ?? message ?? response.body}',
    );
  }
}

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  static const String _historyKey = 'scan_history_v2';

  final ImagePicker _picker = ImagePicker();

  File? _selectedImage;
  PredictionResult? _result;
  bool _isLoading = false;
  int _loadingSeconds = 0;
  List<ScanHistoryItem> _history = [];

  @override
  void initState() {
    super.initState();
    _loadHistory();
  }

  Future<void> _loadHistory() async {
    final prefs = await SharedPreferences.getInstance();
    final savedList = prefs.getStringList(_historyKey) ?? [];

    final loadedHistory = savedList
        .map((item) {
      try {
        return ScanHistoryItem.fromJson(
          jsonDecode(item) as Map<String, dynamic>,
        );
      } catch (_) {
        return null;
      }
    })
        .whereType<ScanHistoryItem>()
        .toList();

    if (!mounted) return;
    setState(() {
      _history = loadedHistory;
    });
  }

  Future<void> _saveHistory() async {
    final prefs = await SharedPreferences.getInstance();
    final encodedList =
    _history.map((item) => jsonEncode(item.toJson())).toList();
    await prefs.setStringList(_historyKey, encodedList);
  }

  void _startLoadingTimer() {
    _loadingSeconds = 0;
    Future.doWhile(() async {
      await Future.delayed(const Duration(seconds: 1));
      if (!_isLoading || !mounted) return false;
      setState(() => _loadingSeconds++);
      return true;
    });
  }

  Future<void> _showImageSourcePicker() async {
    if (_isLoading) return;

    await showModalBottomSheet<void>(
      context: context,
      showDragHandle: true,
      builder: (context) {
        return SafeArea(
          child: Padding(
            padding: const EdgeInsets.fromLTRB(16, 8, 16, 20),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                const Text(
                  'Select Image Source',
                  style: TextStyle(fontSize: 18, fontWeight: FontWeight.w700),
                ),
                const SizedBox(height: 14),
                ListTile(
                  leading: const Icon(Icons.photo_library_outlined),
                  title: const Text('Choose from Gallery'),
                  onTap: () {
                    Navigator.pop(context);
                    _pickImage(ImageSource.gallery);
                  },
                ),
                ListTile(
                  leading: const Icon(Icons.camera_alt_outlined),
                  title: const Text('Take a Photo'),
                  onTap: () {
                    Navigator.pop(context);
                    _pickImage(ImageSource.camera);
                  },
                ),
              ],
            ),
          ),
        );
      },
    );
  }

  Future<void> _pickImage(ImageSource source) async {
    try {
      final XFile? pickedFile = await _picker.pickImage(
        source: source,
        imageQuality: 95,
      );

      if (pickedFile == null) return;

      setState(() {
        _selectedImage = File(pickedFile.path);
        _result = null;
      });
    } catch (e) {
      _showMessage('Error picking image: $e');
    }
  }

  Future<void> _analyzeImage() async {
    if (_selectedImage == null) {
      _showMessage('Please select an image first.');
      return;
    }

    setState(() {
      _isLoading = true;
      _result = null;
    });

    _startLoadingTimer();

    try {
      final result = await ApiService.predictImage(_selectedImage!);

      final historyItem = ScanHistoryItem(
        imagePath: _selectedImage!.path,
        resultType: result.status,
        prediction: result.prediction,
        confidence: result.confidence,
        filename: result.filename,
        time: DateTime.now(),
        benignScore: result.benignScore,
        invalidScore: result.invalidScore,
        malignantScore: result.malignantScore,
        note: result.note,
        message: result.message,
      );

      if (!mounted) return;
      setState(() {
        _result = result;
        _history.insert(0, historyItem);
      });

      await _saveHistory();
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _result = PredictionResult(
          success: false,
          status: 'error',
          filename: _selectedImage?.path.split('/').last ?? '',
          prediction: '',
          confidence: 0.0,
          benignScore: 0.0,
          invalidScore: 0.0,
          malignantScore: 0.0,
          note: '',
          message: 'Something went wrong while sending the image.',
          error: e.toString(),
        );
      });
      _showMessage('Upload error: $e');
    } finally {
      if (!mounted) return;
      setState(() {
        _isLoading = false;
        _loadingSeconds = 0;
      });
    }
  }

  void _newScan() {
    setState(() {
      _selectedImage = null;
      _result = null;
    });
  }

  void _clearCurrent() {
    setState(() {
      _selectedImage = null;
      _result = null;
    });
  }

  Future<void> _clearHistory() async {
    setState(() {
      _history = [];
    });
    final prefs = await SharedPreferences.getInstance();
    await prefs.remove(_historyKey);
  }

  void _openHistoryItem(ScanHistoryItem item) {
    setState(() {
      _selectedImage = File(item.imagePath);
      _result = PredictionResult(
        success: item.resultType == 'ok',
        status: item.resultType,
        filename: item.filename,
        prediction: item.prediction,
        confidence: item.confidence,
        benignScore: item.benignScore,
        invalidScore: item.invalidScore,
        malignantScore: item.malignantScore,
        note: item.note,
        message: item.message,
        error: '',
      );
    });
  }

  void _showMessage(String message) {
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text(message), duration: const Duration(seconds: 5)),
    );
  }

  bool _isMalignant(String prediction) =>
      prediction.toLowerCase() == 'malignant';

  bool _isBenign(String prediction) => prediction.toLowerCase() == 'benign';

  Color _resultBgColor(String prediction) {
    if (_isMalignant(prediction)) return const Color(0xFFFDECEC);
    if (_isBenign(prediction)) return const Color(0xFFEAF7EE);
    return const Color(0xFFFFF4E5);
  }

  Color _resultTextColor(String prediction) {
    if (_isMalignant(prediction)) return const Color(0xFFB42318);
    if (_isBenign(prediction)) return const Color(0xFF067647);
    return const Color(0xFFB54708);
  }

  String _formatPrediction(String prediction) {
    if (_isMalignant(prediction)) return 'Malignant Detected';
    if (_isBenign(prediction)) return 'Benign Detected';
    if (prediction.toLowerCase() == 'invalid') return 'Invalid Image';
    if (prediction.toLowerCase() == 'uncertain') return 'Uncertain Result';
    return prediction;
  }

  String _riskLabel(String prediction, double confidence) {
    if (_isMalignant(prediction)) {
      if (confidence >= 0.90) return 'High Risk';
      if (confidence >= 0.70) return 'Moderate Risk';
      return 'Low Risk';
    }
    if (_isBenign(prediction)) {
      if (confidence >= 0.90) return 'Low Risk';
      if (confidence >= 0.70) return 'Likely Benign';
      return 'Needs Review';
    }
    if (prediction.toLowerCase() == 'invalid') return 'Unsupported';
    return 'Manual Review';
  }

  Color _riskBadgeColor(String prediction, double confidence) {
    final label = _riskLabel(prediction, confidence);
    switch (label) {
      case 'High Risk':
        return const Color(0xFFB42318);
      case 'Moderate Risk':
        return const Color(0xFFF79009);
      case 'Low Risk':
        return const Color(0xFF067647);
      case 'Likely Benign':
        return const Color(0xFF0BA5EC);
      case 'Needs Review':
      case 'Manual Review':
        return const Color(0xFFB54708);
      case 'Unsupported':
        return const Color(0xFFB42318);
      default:
        return Colors.grey;
    }
  }

  String _loadingMessage() {
    if (_loadingSeconds < 10) return 'Analyzing image, please wait...';
    if (_loadingSeconds < 30) return 'Still analyzing... ($_loadingSeconds s)';
    return 'This is taking longer than usual... ($_loadingSeconds s)';
  }

  String _formatDateTime(DateTime time) {
    final hour = time.hour.toString().padLeft(2, '0');
    final minute = time.minute.toString().padLeft(2, '0');
    final day = time.day.toString().padLeft(2, '0');
    final month = time.month.toString().padLeft(2, '0');
    final year = time.year.toString();
    return '$day/$month/$year  $hour:$minute';
  }

  Widget _buildActionButtons() {
    return SizedBox(
      width: double.infinity,
      child: FilledButton.icon(
        onPressed: _isLoading ? null : _showImageSourcePicker,
        icon: const Icon(Icons.add_photo_alternate_outlined),
        label: const Text('Select Image'),
      ),
    );
  }

  Widget _buildImagePreview() {
    if (_selectedImage != null) {
      return ClipRRect(
        borderRadius: BorderRadius.circular(18),
        child: Stack(
          children: [
            Image.file(
              _selectedImage!,
              height: 250,
              width: double.infinity,
              fit: BoxFit.cover,
            ),
            Positioned(
              right: 10,
              top: 10,
              child: Container(
                padding:
                const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
                decoration: BoxDecoration(
                  color: Colors.black.withOpacity(0.55),
                  borderRadius: BorderRadius.circular(999),
                ),
                child: const Text(
                  'Image Selected',
                  style: TextStyle(
                    color: Colors.white,
                    fontSize: 12,
                    fontWeight: FontWeight.w600,
                  ),
                ),
              ),
            ),
          ],
        ),
      );
    }

    return Container(
      height: 250,
      width: double.infinity,
      decoration: BoxDecoration(
        color: const Color(0xFFE9ECF2),
        borderRadius: BorderRadius.circular(18),
        border: Border.all(color: const Color(0xFFD0D5DD)),
      ),
      child: const Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(Icons.image_outlined, size: 48, color: Colors.black45),
            SizedBox(height: 10),
            Text(
              'No image selected yet',
              style: TextStyle(color: Colors.black54, fontSize: 15),
            ),
            SizedBox(height: 6),
            Text(
              'Choose from gallery or take a photo',
              style: TextStyle(color: Colors.black45, fontSize: 12),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildAnalyzeButtons() {
    return Row(
      children: [
        Expanded(
          child: FilledButton.icon(
            onPressed: _isLoading ? null : _analyzeImage,
            icon: const Icon(Icons.analytics_outlined),
            label: const Text('Analyze Image'),
          ),
        ),
        const SizedBox(width: 10),
        Expanded(
          child: OutlinedButton.icon(
            onPressed: _isLoading ? null : _clearCurrent,
            icon: const Icon(Icons.close),
            label: const Text('Clear'),
          ),
        ),
      ],
    );
  }

  Widget _buildLoadingArea() {
    if (!_isLoading) return const SizedBox.shrink();

    return Column(
      children: [
        const SizedBox(height: 18),
        const CircularProgressIndicator(),
        const SizedBox(height: 10),
        Text(
          _loadingMessage(),
          textAlign: TextAlign.center,
          style: const TextStyle(color: Colors.black54),
        ),
      ],
    );
  }

  Widget _buildResultCard() {
    if (_result == null) return const SizedBox.shrink();

    final result = _result!;

    if (result.status == 'invalid') {
      return Card(
        elevation: 1,
        shape:
        RoundedRectangleBorder(borderRadius: BorderRadius.circular(18)),
        child: Padding(
          padding: const EdgeInsets.all(18),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Row(
                children: [
                  Icon(Icons.warning_amber_rounded,
                      color: Color(0xFFB42318)),
                  SizedBox(width: 8),
                  Text(
                    'Invalid Image',
                    style: TextStyle(
                      fontSize: 20,
                      fontWeight: FontWeight.w700,
                      color: Color(0xFFB42318),
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 14),
              Container(
                padding: const EdgeInsets.symmetric(
                    horizontal: 12, vertical: 10),
                decoration: BoxDecoration(
                  color: const Color(0xFFFDECEC),
                  borderRadius: BorderRadius.circular(14),
                ),
                child: Text(
                  result.message.isNotEmpty
                      ? result.message
                      : 'This image is not a valid breast histopathology image.',
                  style: const TextStyle(
                    color: Color(0xFFB42318),
                    fontWeight: FontWeight.w600,
                    height: 1.4,
                  ),
                ),
              ),
              const SizedBox(height: 14),
              Text(
                'Invalid score: ${(result.invalidScore * 100).toStringAsFixed(2)}%',
                style: const TextStyle(fontSize: 15),
              ),
              const SizedBox(height: 16),
              SizedBox(
                width: double.infinity,
                child: OutlinedButton.icon(
                  onPressed: _newScan,
                  icon: const Icon(Icons.refresh),
                  label: const Text('Try Another Image'),
                ),
              ),
            ],
          ),
        ),
      );
    }

    if (result.status == 'uncertain') {
      return Card(
        elevation: 1,
        shape:
        RoundedRectangleBorder(borderRadius: BorderRadius.circular(18)),
        child: Padding(
          padding: const EdgeInsets.all(18),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Row(
                children: [
                  Icon(Icons.error_outline, color: Color(0xFFB54708)),
                  SizedBox(width: 8),
                  Text(
                    'Uncertain Result',
                    style: TextStyle(
                      fontSize: 20,
                      fontWeight: FontWeight.w700,
                      color: Color(0xFFB54708),
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 14),
              Container(
                padding: const EdgeInsets.symmetric(
                    horizontal: 12, vertical: 10),
                decoration: BoxDecoration(
                  color: const Color(0xFFFFF4E5),
                  borderRadius: BorderRadius.circular(14),
                ),
                child: Text(
                  result.message.isNotEmpty
                      ? result.message
                      : 'The model is not confident enough in this image.',
                  style: const TextStyle(
                    color: Color(0xFFB54708),
                    fontWeight: FontWeight.w600,
                    height: 1.4,
                  ),
                ),
              ),
              const SizedBox(height: 14),
              Text(
                'Confidence: ${(result.confidence * 100).toStringAsFixed(2)}%',
                style: const TextStyle(fontSize: 15),
              ),
              const SizedBox(height: 8),
              Text(
                'Benign: ${(result.benignScore * 100).toStringAsFixed(2)}% | '
                    'Malignant: ${(result.malignantScore * 100).toStringAsFixed(2)}%',
                style:
                const TextStyle(fontSize: 14, color: Colors.black54),
              ),
              const SizedBox(height: 16),
              SizedBox(
                width: double.infinity,
                child: OutlinedButton.icon(
                  onPressed: _newScan,
                  icon: const Icon(Icons.refresh),
                  label: const Text('New Scan'),
                ),
              ),
            ],
          ),
        ),
      );
    }

    if (result.status == 'error') {
      return Card(
        elevation: 1,
        shape:
        RoundedRectangleBorder(borderRadius: BorderRadius.circular(18)),
        child: Padding(
          padding: const EdgeInsets.all(18),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Row(
                children: [
                  Icon(Icons.cloud_off, color: Color(0xFFB42318)),
                  SizedBox(width: 8),
                  Text(
                    'Request Error',
                    style: TextStyle(
                      fontSize: 20,
                      fontWeight: FontWeight.w700,
                      color: Color(0xFFB42318),
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 14),
              Text(
                result.message.isNotEmpty
                    ? result.message
                    : 'Could not complete the request.',
                style: const TextStyle(fontSize: 15),
              ),
              if (result.error.isNotEmpty) ...[
                const SizedBox(height: 10),
                Text(
                  result.error,
                  style: const TextStyle(
                      fontSize: 13, color: Colors.black54),
                ),
              ],
            ],
          ),
        ),
      );
    }

    final badgeColor = _riskBadgeColor(result.prediction, result.confidence);

    return Card(
      elevation: 1,
      shape:
      RoundedRectangleBorder(borderRadius: BorderRadius.circular(18)),
      child: Padding(
        padding: const EdgeInsets.all(18),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              'Prediction Result',
              style: TextStyle(fontSize: 20, fontWeight: FontWeight.w700),
            ),
            const SizedBox(height: 14),
            Wrap(
              spacing: 8,
              runSpacing: 8,
              children: [
                Container(
                  padding: const EdgeInsets.symmetric(
                      horizontal: 12, vertical: 8),
                  decoration: BoxDecoration(
                    color: _resultBgColor(result.prediction),
                    borderRadius: BorderRadius.circular(999),
                  ),
                  child: Text(
                    _formatPrediction(result.prediction),
                    style: TextStyle(
                      color: _resultTextColor(result.prediction),
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                ),
                Container(
                  padding: const EdgeInsets.symmetric(
                      horizontal: 12, vertical: 8),
                  decoration: BoxDecoration(
                    color: badgeColor.withOpacity(0.12),
                    borderRadius: BorderRadius.circular(999),
                    border:
                    Border.all(color: badgeColor.withOpacity(0.35)),
                  ),
                  child: Text(
                    _riskLabel(result.prediction, result.confidence),
                    style: TextStyle(
                      color: badgeColor,
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 16),
            Text(
              'Confidence: ${(result.confidence * 100).toStringAsFixed(2)}%',
              style: const TextStyle(fontSize: 15),
            ),
            const SizedBox(height: 8),
            Text(
              'Filename: ${result.filename}',
              style: const TextStyle(fontSize: 15),
            ),
            const SizedBox(height: 8),
            Text(
              'Benign score: ${result.benignScore.toStringAsFixed(4)}',
              style: const TextStyle(fontSize: 15),
            ),
            const SizedBox(height: 8),
            Text(
              'Malignant score: ${result.malignantScore.toStringAsFixed(4)}',
              style: const TextStyle(fontSize: 15),
            ),
            const SizedBox(height: 10),
            Text(
              result.note,
              style: const TextStyle(
                fontSize: 13,
                color: Colors.black54,
                height: 1.4,
              ),
            ),
            const SizedBox(height: 16),
            SizedBox(
              width: double.infinity,
              child: OutlinedButton.icon(
                onPressed: _newScan,
                icon: const Icon(Icons.refresh),
                label: const Text('New Scan'),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildHistoryCard(ScanHistoryItem item) {
    Color chipBg;
    Color chipText;
    String chipTextLabel;

    switch (item.resultType) {
      case 'invalid':
        chipBg = const Color(0xFFFDECEC);
        chipText = const Color(0xFFB42318);
        chipTextLabel = 'Invalid';
        break;
      case 'uncertain':
        chipBg = const Color(0xFFFFF4E5);
        chipText = const Color(0xFFB54708);
        chipTextLabel = 'Uncertain';
        break;
      default:
        final isMalignant = _isMalignant(item.prediction);
        chipBg = isMalignant
            ? const Color(0xFFFDECEC)
            : const Color(0xFFEAF7EE);
        chipText = isMalignant
            ? const Color(0xFFB42318)
            : const Color(0xFF067647);
        chipTextLabel = isMalignant ? 'Malignant' : 'Benign';
    }

    return InkWell(
      borderRadius: BorderRadius.circular(14),
      onTap: () => _openHistoryItem(item),
      child: Card(
        elevation: 1,
        shape:
        RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
        margin: const EdgeInsets.only(bottom: 10),
        child: Padding(
          padding: const EdgeInsets.all(12),
          child: Row(
            children: [
              ClipRRect(
                borderRadius: BorderRadius.circular(10),
                child: Image.file(
                  File(item.imagePath),
                  width: 64,
                  height: 64,
                  fit: BoxFit.cover,
                  errorBuilder: (context, error, stackTrace) {
                    return Container(
                      width: 64,
                      height: 64,
                      color: const Color(0xFFE9ECF2),
                      child: const Icon(Icons.image_not_supported),
                    );
                  },
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Wrap(
                      crossAxisAlignment: WrapCrossAlignment.center,
                      spacing: 8,
                      runSpacing: 6,
                      children: [
                        Text(
                          chipTextLabel,
                          style: const TextStyle(
                            fontSize: 15,
                            fontWeight: FontWeight.w700,
                          ),
                        ),
                        Container(
                          padding: const EdgeInsets.symmetric(
                            horizontal: 8,
                            vertical: 4,
                          ),
                          decoration: BoxDecoration(
                            color: chipBg,
                            borderRadius: BorderRadius.circular(999),
                          ),
                          child: Text(
                            item.resultType,
                            style: TextStyle(
                              fontSize: 11,
                              fontWeight: FontWeight.w700,
                              color: chipText,
                            ),
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 4),
                    Text(
                      'Confidence: ${(item.confidence * 100).toStringAsFixed(2)}%',
                      style: const TextStyle(
                        fontSize: 13,
                        color: Colors.black54,
                      ),
                    ),
                    const SizedBox(height: 2),
                    Text(
                      _formatDateTime(item.time),
                      style: const TextStyle(
                        fontSize: 12,
                        color: Colors.black45,
                      ),
                    ),
                    const SizedBox(height: 4),
                    const Text(
                      'Tap to view result',
                      style: TextStyle(
                        fontSize: 11,
                        color: Colors.black38,
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildHistorySection() {
    return Column(
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            const Text(
              'Scan History',
              style: TextStyle(fontSize: 20, fontWeight: FontWeight.w700),
            ),
            if (_history.isNotEmpty)
              TextButton(
                onPressed: _clearHistory,
                child: const Text(
                  'Clear history',
                  style: TextStyle(color: Color(0xFFB42318)),
                ),
              ),
          ],
        ),
        const SizedBox(height: 8),
        if (_history.isEmpty)
          Card(
            elevation: 1,
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(16),
            ),
            child: const Padding(
              padding: EdgeInsets.all(16),
              child: Text(
                'No scans yet. Your results will appear here.',
                style: TextStyle(color: Colors.black54),
              ),
            ),
          )
        else
          ..._history.map(_buildHistoryCard),
      ],
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Image.asset(
              'assets/images/logo.png',
              width: 30,
              height: 30,
            ),
            const SizedBox(width: 10),
            const Text('Breast Cancer Detection'),
          ],
        ),
        centerTitle: true,
      ),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          const SizedBox(height: 8),
          const Text(
            'Upload a histopathology image for AI-based analysis',
            textAlign: TextAlign.center,
            style: TextStyle(fontSize: 15, color: Colors.black54),
          ),
          const SizedBox(height: 20),
          Card(
            elevation: 1,
            shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(18)),
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                children: [
                  _buildActionButtons(),
                  const SizedBox(height: 16),
                  _buildImagePreview(),
                  const SizedBox(height: 16),
                  _buildAnalyzeButtons(),
                  _buildLoadingArea(),
                ],
              ),
            ),
          ),
          const SizedBox(height: 18),
          _buildResultCard(),
          const SizedBox(height: 20),
          _buildHistorySection(),
          const SizedBox(height: 24),
        ],
      ),
    );
  }
}
